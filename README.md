\# Spaced-Repetition-LoRA



A fine-tuned educational agent based on DeepSeek-R1-7B. This project integrates the Ebbinghaus forgetting curve strategy to enable dynamic, time-aware study guidance.



\## Project Overview



Traditional study methods suffer from two main issues: knowledge points are highly fragmented, and static notes cannot test actual memory retention dynamically. 



This project extracts cross-disciplinary notes and fine-tunes a large language model to adjust its teaching strategy based on the specific review day. On Day 1, the model acts as a tutor providing detailed explanations and breaking down complex concepts. By Day 7, it transforms into an examiner, generating concise quizzes and fill-in-the-blank questions to actively combat the forgetting curve.



\## Performance Benchmark



The model was evaluated using strict automated NLP matching for instruction following and semantic reconstruction.



\- Format Compliance Rate: Improved from 87.5% (base model) to 100%.

\- Strategy Divergence Rate: Day 7 quiz trait generation reached 100%, compared to 15% for the base model.

\- Semantic Distance: The cosine similarity between Day 1 and Day 7 outputs is 0.35, demonstrating two completely independent cognitive logics based on the given day prompt.



\## Quick Start



This repository is fully optimized for native Windows environments with 8GB VRAM, utilizing paged\_adamw\_8bit for memory offloading to prevent OOM errors.



1\. Install dependencies:

pip install torch transformers peft datasets trl



2\. Run the training script:

python train\_native.py



3\. Run the batch evaluation:

python batch\_test.py



\## FAQ



Q: Why choose DeepSeek-R1-7B for fine-tuning?

A: The core objective of this project is to output domain-specific educational strategies rather than general conversational abilities. DeepSeek offers excellent Chinese comprehension, and the 7B parameter scale significantly lowers the training threshold for individual developers, making it viable on consumer-grade GPUs.



Q: Why is there no frontend UI?

A: The primary goal of this project is to explore the algorithmic feasibility of combining LLMs with spaced repetition curves and to provide a high-quality dataset. For application-layer development, I highly encourage the open-source community to fork the repository and submit pull requests.

