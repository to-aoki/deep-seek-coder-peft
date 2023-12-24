from datasets import load_dataset, Dataset


def load_ja_stackoverflow_dataset():
    dataset = load_dataset("p1atdev/ja-stackoverflow", name="simple", split="train")

    instructions = []
    outputs = []
    for r in dataset:
        question_score = r['question_score']
        if question_score is not None and question_score < 0:
            # 負スコア質問は除外する
            continue

        accepted_answer_score = r['accepted_answer_score']
        popular_answer_score = r['popular_answer_score']
        if accepted_answer_score is not None and accepted_answer_score > 0:
            # 受領返答があれば採用
            instructions.append(r['question_body'])
            outputs.append(r['accepted_answer_body'])
        elif popular_answer_score is not None and popular_answer_score > 0:
            # 受領返答がない場合は採用
            instructions.append(r['question_body'])
            outputs.append(r['popular_answer_body'])
        else:
            # 返答なし
            continue

    return Dataset.from_dict(
        {
            "instruction": instructions,
            "output": outputs
        }
    )

ja_stackoverflow_dataset = load_ja_stackoverflow_dataset()
ja_stackoverflow_dataset.to_json("ja_stackoverflow_dataset.json")
raw_train_datasets = load_dataset(
    'json',
    data_files="ja_stackoverflow_dataset.json",
    split="train",
)
print(raw_train_datasets)