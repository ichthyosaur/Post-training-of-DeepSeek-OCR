import random
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Sequence, Value, Image
from typing import List, Dict, Optional
from PIL import Image as PILImage

TARGET_FEATURES = Features({
    "image": Image(decode=True),
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string")
        }
    ],
    "valid": Value("bool")
})

IMAGE_TOKEN = "<image>"
MAX_IMAGE_SIZE = 512 

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

def _resize_image(image):
    MAX_IMAGE_SIZE = 512 
    if image is None: return None
    w, h = image.size
    print(f"Original image size: {w}x{h}")
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        scale = min(MAX_IMAGE_SIZE / w, MAX_IMAGE_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), resample=PILImage.LANCZOS)
    return image

def _format_docvqa(example: Dict) -> Dict:
    try:
        img = example['image']
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _resize_image(img)
            
        question = example['question']
        answer = random.choice(example['answers']) if example['answers'] else ""
        user_content = _get_random_prompt(DOCVQA_TEMPLATES, question=question)
        
        return {
            "image": img,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ],
            "valid": True
        }
    except Exception as e:
        return {
            "image": None, 
            "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
            "valid": False
        }

def _format_coco(example: Dict) -> Dict:
    try:
        if 'image' not in example: raise KeyError("Image column missing")
            
        img = example['image']
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _resize_image(img) 
        
        if 'caption' in example:
            target_caption = example['caption']
        elif 'captions' in example:
            target_caption = random.choice(example['captions'])
        elif 'sentences' in example:
            target_caption = random.choice(example['sentences'])['raw']
        else:
            target_caption = "An image containing various objects."
        
        user_content = _get_random_prompt(COCO_TEMPLATES)
        
        return {
            "image": img,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_caption}
            ],
            "valid": True
        }
    except Exception as e:
        print("failed")
        return {
            "image": None, 
            "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], 
            "valid": False
        }


def build_deepseek_ocr_sft_dataset(
    docvqa_path: str = "lmms-lab/DocVQA",
    coco_path: str =  "lmms-lab/COCO-Caption",
    split: str = "train",
    max_samples: Optional[int] = None,
    num_proc: int = 1,
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
    
    
    ds_docvqa = load_dataset(docvqa_path, "DocVQA", split='validation')
    
    print("Loading COCO (this might take a while for the first time)...")
    # ds_coco = load_dataset(coco_path, split=split)

    if max_samples is not None:
        print(f"Subsampling to {max_samples} samples per dataset...")
        ds_docvqa = ds_docvqa.select(range(min(len(ds_docvqa), max_samples)))
        # ds_coco = ds_coco.select(range(min(len(ds_coco), max_samples)))

    print("Formatting DocVQA data...")
    processed_docvqa = ds_docvqa.map(
        _format_docvqa,
        remove_columns=ds_docvqa.column_names,
        features=TARGET_FEATURES,
        num_proc=num_proc,
        load_from_cache_file=False,
        writer_batch_size=500,
        desc="Processing DocVQA"
    )
    """
    print("Formatting COCO data...")
    processed_coco = ds_coco.map(
        _format_coco,
        remove_columns=ds_coco.column_names,
        features=TARGET_FEATURES,
        num_proc=num_proc,
        writer_batch_size=500,
        desc="Processing COCO"
    )
    processed_coco = processed_coco.filter(lambda x: x['valid'] is True, num_proc=num_proc)
    print(f"DEBUG: DocVQA Valid Samples: {len(processed_docvqa)}")
    print(f"DEBUG: COCO Valid Samples: {len(processed_coco)}")
    print("Merging and shuffling datasets...")

    

    print("Merging datasets...")
    processed_docvqa = processed_docvqa.remove_columns(["valid"])
    processed_coco = processed_coco.remove_columns(["valid"])
    """
    combined_dataset = concatenate_datasets([processed_docvqa])
    combined_dataset = combined_dataset.shuffle(seed=seed)
    
    print(f"Dataset build complete. Total samples: {len(combined_dataset)}")
    return combined_dataset
