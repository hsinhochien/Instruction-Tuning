import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import argparse
from utils import get_bnb_config, get_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for LLM")
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--adapter_checkpoint', required=True, help='Path to adapter_checkpoint')
    parser.add_argument('--input', required=True, help='Path to input.jsonl')
    parser.add_argument('--output', required=True, help='Path to output.jsonl')
    return parser.parse_args()

args = parse_args()

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def preprocess_for_inference(input_text):
    # Tokenize the input (instruction) and add special tokens
    tokenized_instruction = tokenizer(get_prompt(input_text), truncation=True, padding=False)

    # Add special tokens: bos_token_id at the start and a custom special token at the end
    instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instruction["input_ids"] + [tokenizer.eos_token_id]  # 使用特殊標記結束

    # Ensure the length does not exceed max_length
    max_length = 2048
    instruction_input_ids = torch.tensor(instruction_input_ids[:max_length]).unsqueeze(0).to(device)  # Add batch dimension

    return {
        "input_ids": instruction_input_ids
    }

def inference(input_text):
    # 預處理輸入
    inputs = preprocess_for_inference(input_text)
    input_ids = inputs["input_ids"]

    # 使用 model.generate 進行推理
    generated_ids = model.generate(input_ids, max_length=2048, num_return_sequences=1)

    # 先將生成的 token_ids 解碼為文本
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 找到 'ASSISTANT' 的位置
    assistant_index = full_text.find('ASSISTANT')

    # 檢查 'ASSISTANT' 是否存在於生成文本中
    if assistant_index != -1:
        # 提取 'ASSISTANT' 之後的文本
        predicted_text = full_text[assistant_index + len('ASSISTANT'):].strip()
    else:
        # 如果找不到 'ASSISTANT'，返回整個文本
        predicted_text = full_text.strip()

    return predicted_text

private_test = load_json(args.input)
examples_prompt = "範例 1: 翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案： 雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。\n範例 2: 沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文 後未旬，果見囚執。\n範例 3: 辛未，命吳堅為左丞相兼樞密使，常楙參知政事。\n把這句話翻譯成現代文。 初五，命令吳堅為左承相兼樞密使，常增為參知政事。\n範例 4: 十八年，奚、契丹侵犯邊界，以皇上為河北道元帥，信安王為副，率禦史大夫李朝隱、京兆尹裴亻由先等八總管討伐他們。\n翻譯成文言文： 十八年，奚、契丹犯塞，以上為河北道元帥，信安王禕為副，帥禦史大夫李朝隱、京兆尹裴伷先等八總管兵以討之。\n範例 5: 正月，甲子朔，鼕至，太後享通天宮；赦天下，改元。\n把這句話翻譯成現代文。 聖曆元年正月，甲子朔，鼕至，太後在通天宮祭祀；大赦天下，更改年號。\n範例 6: 文言文翻譯：\n明日，趙用賢疏入。 答案：第二天，趙用賢的疏奏上。\n範例 7: 我當時在三司，訪求太祖、仁宗的手書敕令沒有見到，然而人人能傳誦那些話，禁止私鹽的建議也最終被擱置。\n翻譯成文言文： 餘時在三司，求訪兩朝墨敕不獲，然人人能誦其言，議亦竟寢。\n範例 8: 娶上榖公主，被授予駙馬都尉。\n這句話在古代怎麼說： 尚上榖公主，拜駙馬都尉。\n範例 9: 後二年，移鄧州，又徙襄州。\n把這句話翻譯成文言文： 後二年，徙鄧州，又徙襄州。\n範例 10: 令、錄、簿、尉等職官有年老病重的人允許彈勃。\n翻譯成文言文： 令、錄、簿、尉諸職官有耄耋篤疾者舉劾之。\n範例 11: 將下麵句子翻譯成文言文：\n而且姑侄與母子相比誰更親？ 且姑侄之與母子孰親？\n範例 12: 翻譯成現代文：\n州民鄭五醜構逆，與叛羌傍乞鐵匆相應，令剛往鎮之。\n答案： 渭州人鄭五醜造反，與叛逆羌傍乞鐵忽互相呼應。下令趟剛前往鎮壓。\n範例 13: 翻譯成現代文：\n士匄請見，弗內。\n答案： 士匄請求進見，荀偃不接見。\n範例 14: 翻譯成文言文：\n富貴貧賤都很尊重他。\n答案： 貴賤並敬之。\n範例 15: 翻譯成文言文：\n到春天長齣青草的時候，瓜也同時齣土，瓜苗莖葉肥壯茂盛，勝過通常的瓜苗。 至春草生，瓜亦生，莖葉肥茂，異於常者。\n範例 16: 將下麵句子翻譯成現代文：\n昨日破城，將士輕敵，微有不利，何足為懷。 昨日破城時，將士輕敵，隻有一點小小的不利，何必掛在心上？\n範例 17: 翻譯成文言文：\n龐勛率領軍隊二萬人從石山嚮西進發，所過之處燒殺搶掠，一無所存。\n答案： 龐勛將兵二萬自石山西齣，所過焚掠無遺。\n範例 18: 議雖不從，天下鹹重其言。\n翻譯成白話文： 他的建議雖然不被采納，但天下都很敬重他的話。\n範例 19: 他請求退休，但下詔不許。\n翻譯成文言文： 求緻仕，詔不許。\n範例 20: 將下麵句子翻譯成文言文：\n第二年，徐溫冊立楊渭為天子，僭號稱為大吳國，改唐朝天祐十六年為武義元年。 明年，溫冊楊渭為天子，僭稱大吳，改唐天祐十六年為武義元年。"

# Add examples to "instruction"
for item in private_test:
    item["instruction"] = f"{examples_prompt}\n指令: {item['instruction']}"

# Define tokenizer and model
model_name = args.model
adapter_checkpoint = args.adapter_checkpoint
bnb_config = get_bnb_config()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})

# Load the fine-tuned adapter checkpoint
model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
model.config.use_cache = False
model.eval()

# Move model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 存儲預測結果
predictions = []

# 遍歷測試資料進行推理
for sample in tqdm(private_test, desc="Predicting", unit="sample"):
    input_text = sample["instruction"]
    predicted_output = inference(input_text)

    # 後處理：檢查並移除第一個字元是冒號的情況
    if predicted_output.startswith(":"):
        predicted_output = predicted_output[1:].strip()

    # 將結果保存為字典
    predictions.append({
        "id": sample["id"],
        "output": predicted_output
    })

# 保存預測結果到新的 JSON 文件
save_json(predictions, args.output)
print(f"Results saved to {args.output}")