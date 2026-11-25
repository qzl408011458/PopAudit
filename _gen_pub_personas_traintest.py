import os
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# Qwen3-8B generation settings (sampling is not seeded)
# =========================

PUB_DATA = 'pums'
# PUB_DATA = 'NHTS'

def convert_to_12_hour_clock(minutes):
    """
    Convert minutes since midnight (0–1439) to 12-hour clock format string.
    Example:
        510 -> 8:30 AM
        0   -> 12:00 AM
        720 -> 12:00 PM
    """
    try:
        minutes = int(minutes)
    except (ValueError, TypeError):
        return str(minutes)

    hours = minutes // 60
    mins = minutes % 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    hour_12 = hours % 12
    if hour_12 == 0:
        hour_12 = 12

    return f"{hour_12}:{mins:02d} {period}"

class Persona_Generator:
    def __init__(self, device="cuda:0"):
        model_name = "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16
        )

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # keep same "skip </think>" attempt logic
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("content:", content)
        return content


# =========================
# Final variable grouping (your confirmed classification)
# =========================
if PUB_DATA == 'pums':
    DEMO_COLS = [
        "Region_Category",
        "Gender",
        "Age",
        "Marital_status",
        "Employment_status",
        "Studenthood_status",
        "Personal_Income_Class",
    ]

    HOUSEHOLD_COLS = [
        "Household_Type",
        "Household_Size",
        "Number_of_children",
        "Number_of_cars_of_household",
        "Household_Income_Class",
    ]

    BEHAVIOR_COLS = [
        "Commute_Mode",
        "Commute_Travel_Time_Minutes",
        "Commute_Arrival_Time_Minutes"
    ]

if PUB_DATA == 'NHTS':
    DEMO_COLS = [
        "Municipality_Categories",
        "Gender",
        "Age",
        "Employment_status",
        "Studenthood_status",
        "Income_class",
    ]

    HOUSEHOLD_COLS = [
        "Household_Type",
        "Household_Size",
        "Number_of_children",
        "Number_of_cars_of_household",
    ]

    BEHAVIOR_COLS = [
        "Trips_of_Car",
        "Trips_of_PublicTransport",
        "Trips_of_Walking",
        "Trips_of_Bike",
        "Activities_of_Home",
        "Activities_of_Work",
        "Activities_of_Other",
        "Activities_of_School",
    ]




def _maybe_restore_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the CSV stored an old index column like 'Unnamed: 0', restore it as df.index.
    This is important if dataset_ids.pkl stores "original index" values.
    """
    if df.shape[1] == 0:
        return df
    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed"):
        s = pd.to_numeric(df[first_col], errors="coerce")
        if s.notna().mean() > 0.99 and s.nunique() == len(df):
            df = df.set_index(first_col)
    return df


def load_split_indices(pkl_path: str):
    """
    Supports:
      - dict: {'train_idx':..., 'val_idx':..., 'test_idx':...}
      - tuple/list: (train_idx, test_idx) or (train_idx, val_idx, test_idx)
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        train_idx = obj.get("train_idx") or obj.get("train_ids") or obj.get("train")
        val_idx = obj.get("val_idx") or obj.get("val_ids") or obj.get("val")
        test_idx = obj.get("test_idx") or obj.get("test_ids") or obj.get("test")
        if train_idx is None or test_idx is None:
            raise ValueError("pickle dict does not contain train/test indices.")
        return list(train_idx), (list(val_idx) if val_idx is not None else None), list(test_idx)

    if isinstance(obj, (list, tuple)):
        if len(obj) == 2:
            train_idx, test_idx = obj
            return list(train_idx), None, list(test_idx)
        if len(obj) == 3:
            train_idx, val_idx, test_idx = obj
            return list(train_idx), list(val_idx), list(test_idx)

    raise ValueError("Unrecognized dataset_ids.pkl format.")


def _row_to_info_blocks(row: pd.Series):
    """
    Build the three text blocks in *your style* (like demo_info_json, hhould_info_json, beh_info_json),
    but only include columns that exist and are non-null.
    """
    demo_info_json = "Demographic Information:\n"
    hhould_info_json = "Household Information:\n"
    beh_info_json = "Behavioral Information:\n"

    for c in DEMO_COLS:
        if c in row.index and pd.notna(row[c]):
            demo_info_json += f"\t{c}: {row[c]}\n"

    for c in HOUSEHOLD_COLS:
        if c in row.index and pd.notna(row[c]):
            hhould_info_json += f"\t{c}: {row[c]}\n"

    for c in BEHAVIOR_COLS:
        if c in row.index and pd.notna(row[c]):

            if c == "Commute_Arrival_Time_Minutes":
                time_str = convert_to_12_hour_clock(row[c])
                beh_info_json += f"\tCommute_Arrival_Time: {time_str}\n"

            elif c == "Commute_Travel_Time_Minutes":
                beh_info_json += f"\tCommute_Travel_Time: {int(row[c])} minutes\n"

            else:
                beh_info_json += f"\t{c}: {row[c]}\n"

    return demo_info_json, hhould_info_json, beh_info_json


def _build_prompt(
    demo_info_json: str,
    hhould_info_json: str,
    beh_info_json: str,
    use_iot: bool,
    random_per: bool,
):
    """
    This function intentionally mirrors your prompt template logic line-by-line.
    """
    if not random_per:
        prompt = demo_info_json + "\n" + hhould_info_json + "\n" + beh_info_json + "\n"
        prompt += (
            "According to the above demographic, household and behavioral information of a person, "
            "summarize the characteristics of personal profile and travel preferences in one paragraph."
        )
        if use_iot:
            iot = """
    Requirements:
    1. The generated content must not contain exact numbers, which you should use abstract words to conclude the characteristics.
    2. The whole words should be around 100."""
            prompt += iot
        else:
            iot = """
                    Requirements:
    1. The whole words should be around 100."""
            prompt += iot
    else:
        prompt = "Randomly generate a paragraph describing a person within around 100 words."  # random_ctx2

    return prompt


def _save_path_for_type(type_name: str, use_iot: bool, random_per: bool, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not random_per:
        if use_iot:
            return out_dir / f"{type_name}data_pers.csv"
        return out_dir / f"{type_name}data_exact_ctx.csv"
    return out_dir / f"{type_name}data_random_ctx.csv"


def generate_split_csv(
    df: pd.DataFrame,
    indices: list,
    type_name: str,
    generator: Persona_Generator | None,
    out_dir: str = "data",
    use_iot: bool = False,
    random_per: bool = False,
    gen_persona: bool = True,
):
    data_list = []

    sub = df.loc[indices].copy()

    total_n = len(sub)

    # wrap iterrows with tqdm
    iterator = tqdm(
        sub.iterrows(),
        total=total_n,
        desc=f"Generating {type_name} personas",
        dynamic_ncols=True,
        leave=True
    )

    for pid, row in iterator:
        start_time = time.time()

        data_dic_i = {}
        data_dic_i["Pid"] = pid

        for c in (DEMO_COLS + HOUSEHOLD_COLS + BEHAVIOR_COLS):
            if c in row.index:
                data_dic_i[c] = row[c]

        if gen_persona:
            demo_info_json, hhould_info_json, beh_info_json = _row_to_info_blocks(row)
            prompt = _build_prompt(
                demo_info_json,
                hhould_info_json,
                beh_info_json,
                use_iot,
                random_per,
            )

            content = generator.generate(prompt)

            data_dic_i["prompt"] = prompt
            data_dic_i["persona"] = content

            # show the time of the current row to the right of the progress bar
            iterator.set_postfix(
                last_time=f"{time.time() - start_time:.2f}s"
            )

        data_list.append(data_dic_i)

    df_save = pd.DataFrame(data_list)

    out_path = _save_path_for_type(type_name, use_iot, random_per, out_dir)
    df_save.to_csv(out_path, index=False)

    print(f"[{type_name}] saved -> {out_path} | rows={len(df_save)}")


def main(
    csv_path="data/person_level_merged_compact.csv",
    ids_pkl_path="data/dataset_ids.pkl",
    out_dir="data",
    device="cuda:0",
    use_iot=False,
    random_per=False,
    gen_test_persona=False,
    gen_test_set=False
):
    # load full dataset
    df = pd.read_csv(csv_path)
    df = _maybe_restore_index(df)

    train_idx, val_idx, test_idx = load_split_indices(ids_pkl_path)

    generator = None
    generator = Persona_Generator(device=device)

    # train: always generate persona
    generate_split_csv(
        df=df,
        indices=train_idx,
        type_name="train",
        generator=generator,
        out_dir=out_dir,
        use_iot=use_iot,
        random_per=random_per,
        gen_persona=True,
    )

    # val: generate persona if val_idx exists
    if val_idx is not None and len(val_idx) > 0:
        generate_split_csv(
            df=df,
            indices=val_idx,
            type_name="val",
            generator=generator,
            out_dir=out_dir,
            use_iot=use_iot,
            random_per=random_per,
            gen_persona=True,
        )
    else:
        print("[val] no val indices in dataset_ids.pkl -> skipped.")

    # test: by default no persona (fast + aligns with many pipelines)
    if gen_test_set:
        generate_split_csv(
            df=df,
            indices=test_idx,
            type_name="test",
            generator=generator if gen_test_persona else None,
            out_dir=out_dir,
            use_iot=use_iot,
            random_per=random_per,
            gen_persona=gen_test_persona,
        )


if __name__ == "__main__":
    # ---- Set your ablation mode here (exactly like paper) ----
    # Mode A (abstract/pers): use_iot=True, random_per=False
    # Mode B (exact ctx): use_iot=False, random_per=False
    # Mode C (random ctx): use_iot=False, random_per=True
    use_iot = True
    random_per = False
    device = "cuda:1"

    main(
        csv_path=f"data/{PUB_DATA}/person_level_merged_compact.csv",
        ids_pkl_path=f"data/{PUB_DATA}/dataset_ids.pkl",
        out_dir=f"data/{PUB_DATA}",
        device=device,
        use_iot=use_iot,
        random_per=random_per,
        gen_test_persona=False,  # keep False unless you explicitly want persona for test
        gen_test_set=True
    )