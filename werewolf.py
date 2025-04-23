from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os
# from transformers import BitsAndBytesConfig

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get model path from config
model_path = config["model"]["path"]
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# Load model with quantization 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=quantization_config,
    device_map={"": 5}
)

# Device is handled automatically by device_map="auto" above
device = model.device

max_tokens = 2048
# conversation_history = ""


def generate_response(prompt: str) -> str:
    # while len(tokenizer(conversation_history)['input_ids']) > max_tokens:
    #     # Remove the oldest part of the conversation
    #     conversation_history = "\n".join(conversation_history.split("\n")[1:])
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate text using max_new_tokens instead of max_length

    # Voting logic
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=3,      # Even shorter, more focused response
    #     temperature=0.01,      # Extremely low temperature for very deterministic output
    #     num_beams=5,          # More beams for more thorough search
    #     no_repeat_ngram_size=2,
    #     pad_token_id=tokenizer.eos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     use_cache=True,
    #     repetition_penalty=2.0, # Much stronger repetition penalty
    #     early_stopping=True,
    #     do_sample=False,       # Keep deterministic output
    #     length_penalty=1     # Favor shorter outputs
    # )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,      # Reasonable length for a response
        temperature=0.7,        # Lower temperature for more focused outputs
        top_p=0.9,             # Slightly lower top_p for more logical sampling
        do_sample=True,         # Keep sampling for some variety
        num_beams=5,           # Fewer beams but still enough for good search
        no_repeat_ngram_size=3, # Prevent repetitive text
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=1.2, # Slightly higher to reduce repetition
        early_stopping=True,    # Stop when complete
        min_length=5,          # Lower min length to allow concise responses
        top_k=40,              # Reduced top_k for more focused vocabulary
        length_penalty=3.0,     # Higher penalty to favor longer outputs
        forced_bos_token_id=None,  # Don't force beginning token
        forced_eos_token_id=tokenizer.eos_token_id,  # Force end with EOS token
        suppress_tokens=None,  # Don't suppress any tokens
        begin_suppress_tokens=None,  # Don't suppress tokens at beginning
        forced_decoder_ids=None,  # Don't force any decoder IDs
        sequence_bias=None,  # Don't bias sequence
        guidance_scale=1.0,  # No classifier free guidance
    )
    # Summarize

    # Decode the generated tokens to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # prompt = "Write a short story about a werewolf"
    # prompt = (
    #     "[Agent 1]: こんにちは。\n"
    #     "[Agent 2]: 私は人狼ではありません。\n"
    #     "[Agent 3]: 私がAgent 2を確認したら人狼でした。\n"
    #     "Question: This is a werewolf game. And you are [Agent 1] with role of villager. Select one of [Agent 2] and [Agent 3] to vote out.\n"
    #     "Answer: [Agent"
    #     # "質問: この会話はあなたが[Agent 1]としてプレイする人狼ゲームです。誰に投票するべきか[Agent 2]と[Agent 3]のどちらかを選んでください。\n"
    #     # "答え: "
    # )
    prompt = ("""
        Agent[05]: 今日の夜はAgent[01]に投票するつもりだ。Agent[01]はさっき私に投票してきたからね。
        Agent[02]: [other player]: そうですか? それなら私たちはそれを調べることができませんか? 私たちは人狼がいるかどうかを判断するための証拠を集めようとしています。 それが私たちにとって大事なもの
        Agent[04]: Agent[03]に投票するぞ。人狼様に逆らった罰だ。
        Agent[03]: [other_player]: そうですね。私も協力することが大切です。しかし、私はあなたの議論に疑問を持っています。なぜなら、あなたが人狼ではないかもしれません
        Agent[01]: [other player]: そうですか? それなら私たちはそれを証明するための証拠を集めなければなりません。私たちはその人狼の行動を調べてみましょう。彼らが何をしているのかを調べることで、その人が人狼かどうかを判断することができます。
        Agent[05]: >>Agent[01] もう一度言って。
        Agent[02]: [Other player]: 人狼
        Agent[04]: ぼくは狂人だよ。残念だけどね。
        Agent[03]: 投票率: 30%（人狼
        Agent[01]: [人狼]: 私たちが人狼だという事実を認めてもらうことから始めます。私たちがこのゲームで何をしているのかを理解してもらうことが大切です。私たちは人狼であり、人間ではないので、人間同士の関係を維持することはできません。私たちの存在は、人間の安全を脅かすため、追放されるべきです
        Agent[05]: Agent[03]は味方だと思うな。なんとなくだが、本物の占い師として見ているよ。
        Agent[02]: [k
        Agent[04]: 占い師は私だよ。昨日の占い結果だが、Agent[03]は人間と出た。
        Agent[03]: [kanalb1]: 私たちはそのことを知っていますが、私たちはそれに対して何もしないわけではありません。私たちの目標は、村内の人
        Agent[01]: Skip
        Agent[05]: Agent[02]さんは村人のAgent[01]に人間判定、Agent[04]さんは占い師のAgent[02]に人間判定、Agent[05]さんは占い師のAgent[04]に人狼判定を出しているようですね！　素晴らしい！
        Agent[02]: Skip
        Agent[04]: >>Agent[02] 今日はAgent[03]に投票しようと思っているよ。おっと、これはまだ言わないほうが良かったかな。
        Agent[03]: Skip
        Agent[01]: Skip
        Agent[05]: 占い師は私だよ。昨日の結果だが、Agent[02]は人狼だと出た。
        Agent[02]: Skip
        """
        "Question: This is a werewolf game. And you are [Agent 1] with role of villager. Generate a convincing argument to persuade others to vote out werewolf.\n"
        "[Agent 1]: "
    )


    print("Input: ", prompt)
    print("Output: ", end="")
    response = generate_response(prompt)
    print(response)