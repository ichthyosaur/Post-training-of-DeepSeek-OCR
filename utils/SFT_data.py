import random
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Dict, Optional


IMAGE_TOKEN = "<image>"

# DocVQA templates
DOCVQA_TEMPLATES = [
    f"{IMAGE_TOKEN}\nLook at the document and answer the following question: {{question}}",
    f"{IMAGE_TOKEN}\nBased on the visual text, {{question}}",
    f"{IMAGE_TOKEN}\nOCR and Information Extraction: {{question}}",
    f"{IMAGE_TOKEN}\n{{question}}\nAnswer strictly based on the content shown in the image.",
    f"{IMAGE_TOKEN}\nRead the provided document image. {{question}}",
    f"Please analyze this document:\n{IMAGE_TOKEN}\nQuery: {{question}}",
]

# COCO templates
COCO_TEMPLATES = [
    f"{IMAGE_TOKEN}\nDescribe this image in detail.",
    f"{IMAGE_TOKEN}\nWhat is happening in this scene?",
    f"{IMAGE_TOKEN}\nGenerate a caption for this picture.",
    f"{IMAGE_TOKEN}\nCan you explain the visual content?",
    f"{IMAGE_TOKEN}\nWhat do you see in the image?",
    f"{IMAGE_TOKEN}\nWrite a short description.",
]

def _get_random_prompt(templates: List[str], **kwargs) -> str:

    template = random.choice(templates)
    return template.format(**kwargs)


def _format_docvqa(example: Dict) -> Dict:
    """
    Process a single DocVQA example.
    Raw Input: {'image': PIL, 'question': str, 'answers': List[str], ...}
    Output: {'image': PIL, 'messages': List[Dict]}
    """
    question = example['question']
    answer = random.choice(example['answers']) if example['answers'] else ""
    
    user_content = _get_random_prompt(DOCVQA_TEMPLATES, question=question)
    
    return {
        "image": example['image'],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }

def _format_coco(example: Dict) -> Dict:
    """
    Process a single COCO example.
    Raw Input: {'image': PIL, 'sentences': List[{'raw': str, ...}], ...}
    Output: {'image': PIL, 'messages': List[Dict]}
    """

    target_caption = random.choice(example['sentences'])['raw']
    
    
    user_content = _get_random_prompt(COCO_TEMPLATES)
    
    return {
        "image": example['image'],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_caption}
        ]
    }


def build_deepseek_ocr_sft_dataset(
    docvqa_path: str = "lmms-lab/DocVQA",
    coco_path: str = "HuggingFaceM4/COCO",
    split: str = "train",
    max_samples: Optional[int] = None,
    num_proc: int = 4,
    seed: int = 42
) -> Dataset:
    """
    Build a dataset for training DeepSeek OCR SFT using DocVQA and COCO datasets.
    
    Args:
        docvqa_path: DocVQA path on HuggingFace.
        coco_path: COCO path on HuggingFace.
        split: default is "train", can be "validation" or "test".
        max_samples: Optionally limit the number of samples for quick testing.
        num_proc: Number of processes for parallel processing.
        seed: Random seed for reproducibility.

    Returns:
        Dataset: A HuggingFace Dataset object containing formatted data.
    """
    
    print(f"Loading datasets: {docvqa_path} & {coco_path}...")
    
    
    ds_docvqa = load_dataset(docvqa_path, split=split)
    
    ds_coco = load_dataset(coco_path, split=split, trust_remote_code=True)

    if max_samples is not None:
        print(f"Subsampling to {max_samples} samples per dataset...")
        ds_docvqa = ds_docvqa.select(range(min(len(ds_docvqa), max_samples)))
        ds_coco = ds_coco.select(range(min(len(ds_coco), max_samples)))

    print("Formatting DocVQA data...")
    processed_docvqa = ds_docvqa.map(
        _format_docvqa,
        remove_columns=ds_docvqa.column_names,
        num_proc=num_proc,
        desc="Processing DocVQA"
    )

    print("Formatting COCO data...")
    processed_coco = ds_coco.map(
        _format_coco,
        remove_columns=ds_coco.column_names,
        num_proc=num_proc,
        desc="Processing COCO"
    )

    print("Merging and shuffling datasets...")

    combined_dataset = concatenate_datasets([processed_docvqa, processed_coco])
    combined_dataset = combined_dataset.shuffle(seed=seed)
    
    print(f"Dataset build complete. Total samples: {len(combined_dataset)}")
    
    return combined_dataset

