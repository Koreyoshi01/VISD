import json
from torch.utils.data import Dataset as TorchDataset

SYSTEM_PROMPT = {
    "visual QA": "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant must localize the object with bounding box coordinates between <box> and </box>. The answer must strictly follow the following format:`<obj>object_name</obj><box>bounding_box</box>'.",
    "temporal-spatial free-form QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",
    "temporal QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'.The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'.",
    "temporal QA (MCQ)": "A conversation between user and assistant. The user provides a video and a multiple-choice question, and the Assistant determines the precise time period that answers the question and selects the correct option. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s.\nCorrect Option: [ONLY THE LETTER]'.",
    "General video QA MCQ": "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. Only output the correct option in the <answer> </answer> section.",
    "General video QA Free-form": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
}

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}


class SimpleListDataset(TorchDataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def map(self, fn):
        return SimpleListDataset([fn(dict(record)) for record in self.records])

    def select(self, indices):
        return SimpleListDataset([self.records[index] for index in indices])


class SimpleDatasetDict(dict):
    def map(self, fn):
        return SimpleDatasetDict({split: dataset.map(fn) for split, dataset in self.items()})


def _load_json_records(dataset_name):
    if dataset_name.endswith(".jsonl"):
        with open(dataset_name, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with open(dataset_name, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level list in {dataset_name}, but got {type(payload).__name__}.")
    return payload


def _load_json_dataset_hf(dataset_name):
    from datasets import Dataset, DatasetDict

    return DatasetDict({"train": Dataset.from_json(dataset_name)})


def _load_json_dataset_simple(dataset_name):
    return SimpleDatasetDict({"train": SimpleListDataset(_load_json_records(dataset_name))})

def make_conversation_image_and_video(example):

    task = example.get('task')

    if task == 'visual QA':
        system_message = SYSTEM_PROMPT['visual QA']
        content_list = [{"type": "image"}, {"type": "text", "text": example['question']}]
    elif task in ['temporal-spatial free-form QA', 'temporal QA', 'temporal QA (MCQ)', 'General video QA MCQ', 'General video QA Free-form']:
        system_message = SYSTEM_PROMPT[task]
        content_list = [{"type": "video"}, {"type": "text", "text": example['question']}]
    else:
        raise ValueError(f"Unknown task: {task}")

    prompt_list = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": content_list}
    ]

    example['prompt'] = prompt_list   
    return example


def get_data(script_args):
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        json_loader = getattr(script_args, "json_loader", "hf")
        if json_loader == "hf":
            dataset = _load_json_dataset_hf(script_args.dataset_name)
        elif json_loader == "simple":
            dataset = _load_json_dataset_simple(script_args.dataset_name)
        else:
            raise ValueError(f"Unknown json_loader: {json_loader}")
    else:
        from datasets import load_dataset
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    dataset = dataset.map(make_conversation_image_and_video)

    train_dataset = dataset['train']
    num_to_keep = len(train_dataset) - (len(train_dataset) % 4)
    dataset['train'] = train_dataset.select(range(num_to_keep))
    print(f"Dataset 'train' split size: {num_to_keep}")
    print(dataset)

    return dataset
