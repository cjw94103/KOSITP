# 1. Introduction
<p align="center"><img src="https://github.com/cjw94103/KOSITP/assets/45551860/c645a919-5c60-4392-ad3d-9ad4150afa69" width="35%" height="35%"></p>
SLLM은 Smaller Large Language Model의 약자로 큰 언어 모델(Large Language Model) 중에서도 상대적으로 작은 크기를 가진 모델을 의미합니다. 이들은 흔히 말하는 SLM(Small Language Model)보다는 크지만, 최대 규모의 언어 모델에 비해서는 작습니다. 이 모델들은 여전히 대규모 데이터셋을 사용하여 학습되며, 복잡한 언어 이해 및 생성 작업을 수행할 수 있는 능력을 갖추고 있습니다. 기존의 좋은 성능의 언어 모델들을 그 크기가 매우 거대하여 (예를 들어, GPT 3.5의 경우 175B) 개인이 언어 모델을 학습하기 쉽지 않습니다. 이러한 이유로 본 프로젝트는 오픈 소스로 공개되어 있는 1 ~ 13B 사이의 pretrained model을 이용하고 AIHub, Kisti 등 다양한 한국어 데이터셋을 Instruction format으로 변환하여 한국어 대상의 Instruction Tuned 모델을 개발하고 자연스러운 출력을 위하여 DPO 등 강화학습 방법을 구현하여 좋은 출력의 SLLM을 만드는 것을 목표로 합니다. 학습 프레임워크는 메모리 절약, 학습 속도 가속화를 위한 Unsloth Open Source Library를 사용하며 학습된 모델을 VLLM에서 사용할 수 있게 코드로 공개할 예정입니다. 또한 FastAPI를 통해 모델의 추론을 통신하고 Chainlit으로 간단한 홈페이지를 구현하여 웹 상에서의 챗봇을 구현해볼 예정입니다. 업데이트는 비주기적으로 될 예정입니다.

# 2. Update History

# 3. Dataset
데이터셋은 AIHub, Kisti에서 제공한 데이터셋을 사용하며 Instruction Tuning을 위하여 SuperNI(https://github.com/allenai/natural-instructions) 에 정의된 Task를 참고하여 가능한 22개의 Task Dataset으로 Reformatting을 진행하였습니다. 데이터셋을 전부 공개하지 못하여 sample_data 폴더 안에 Task 별 예제 데이터를 업로드하였습니다. 또한 자연어 생성의 자연스러움을 위하여 k_rlhf, every_lm, evolve_instr 데이터셋을 일부 추가하여 학습을 진행하였습니다. 데이터셋별 대략적인 개수는 아래와 같습니다.
- AIHUB, KISTI :
- k_rlhf, every_lm, evolve_instr :
  
Reformatted Task의 대한 설명은 아래와 같습니다.

|Task Name|Description|
|------|-------|
|Answerablilty Classification|질문이 주어지면 그 질문이 문단에서 답할 수 있는지에 대한 여부를 결정|
|Summarization|텍스트가 주어지면 텍스트의 1/3 이상 분량의 summarization 생성|
|Title Generation|텍스트의 일부가 주어지면 알맞는 제목을 생성|
|Abstractive QA Objective|질문이 주어지면 질문에 맞는 답을 주어진 보기에서 선택|
|Abstractive QA Subjective Short answer|질문이 주어지면 질문에 맞는 답을 단답으로 생성|
|Abstractive QA Subjective Long answer|질문이 주어지면 질문에 맞는 답을 길고 자세하게 생성|
|Extractive QA Objective|질문이 주어지면 질문에 맞는 답을 컨텍스트를 참고하여 주어진 보기에서 선택|
|Extractive QA Yes or No|질문이 주어지면 질문에 맞는 답을 컨텍스트를 참고하여 예, 아니오로 대답|
|Extractive QA Subjective Short answer|질문이 주어지면 질문에 맞는 답을 컨텍스트를 참고하여 단답으로 생성|
|Extractive QA Subjective Long answer|질문이 주어지면 질문에 맞는 답을 컨텍스트를 참고하여 길고 자세하게 생성|
|Text Completion|미완성 형태의 텍스트가 주어지면 나머지 부분을 예측|
|Title2Contents Generation|짧은 제목이 주어지면 주어진 제목과 관련된 텍스트를 생성|
|Text Simplification|텍스트가 주어지면 텍스트를 50글자 이하의 문장으로 단순화|
|Keyword Tagging|텍스트 단락이 주어지면 텍스트 단락을 대표하는 다수의 키워드 생성|
|Text Composition|2~4개의 문장이 주어지면 중간의 빈 문장 단락을 생성|
|Summary(Simplification) Expansion|요약된 문장이 주어지면 요약되지 않은 문장으로 재생성|
|Table QA Short answer|HTML 형식의 표 자료가 주어지면 정답을 작성|
|Paraphrasing|텍스트의 의미를 온전히 보존하면서 다른 표현으로 변환하는 텍스트를 생성|
|Table2Title|HTML 형식의 표 자료가 주어지면 표를 대표하는 제목 생성|
|Abstractive QA Yes or No|질문이 주어지면 질문에 맞는 답을 예, 아니오로 생성|
|Extractive QA Objective Explanation|질문이 주어지면 질문에 맞는 답 주어진 보기에서 선택 후 참고 텍스트에서 정답의 근거를 함께 제시|
|Extractive QA Yes or No Explanation|질문이 주어지면 질문에 맞는 답을 예, 아니오로 생성하고 참고 텍스트에서 정답의 근거를 함께 제시|
# 4. Train
# 5. Inference
# 6. 정량적 평가
