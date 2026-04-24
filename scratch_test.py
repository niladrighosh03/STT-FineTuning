from datasets import load_dataset
import os

def test():
    print("Loading with split slicing...")
    train_dataset = load_dataset("librispeech_asr", "clean", split="train.100[:1%]")
    print("Loaded!", len(train_dataset))
    
    os.system("du -sh /workspace/hf_cache/hub")

if __name__ == "__main__":
    test()
