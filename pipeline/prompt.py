PROMPT = """There is a [hidden_type] in the image, what is it ?"""

COT_PROMPT = """You are an expert in solving visual puzzles and optical illusions. Your task is to identify the hidden [hidden_type] embedded in the image. 

The image is designed as an optical illusion, where the character is subtly integrated into the semantic background or noise patterns. To identify the hidden content, you can simulate human visual behaviors:
1. Imagine squinting your eyes or slightly blurring your vision. Ignore the sharp, high frequency details, textures and noise in the image.
2. Imagine viewing the image from a long distance. You can resize the image smaller in your mind to get a global view of the image.
You can combine the two strategies to enhance your perception of the hidden character.

Now, please analyze the image carefully, and identify the hidden [hidden_type].
"""

MULTI_SCALE_PROMPT = """I provide four views of the SAME image, the original view and the global views. There is a SAME [hidden_type] embedded in these images, with the help of the views, what is it ?"""

EVAL_PROMPT = """You are a strict evaluator. Your task is to determine whether the model's response correctly identifies the hidden number(s), letter(s), word(s), or Chinese character(s) in the image.

You will be given a ground truth answer, which is the correct hidden content, and a model response, which is the content identified by a specific model. You should compare the model response with the ground truth answer and decide if the model's identification is correct.
- [Correct]: If the model response exactly matches the ground truth answer.
- [Incorrect]: If the model response does not match the ground truth answer.
Your output should only contain your evaluation result, either "Correct" or "Incorrect", without any additional explanation or commentary.

# Example 1
Ground Truth Answer: 5
Model Response: Looking at the image carefully, I can see it's a cyberpunk-style scene with neon lights and futuristic architecture. However, if I look at the image from a distant perspective, I can identify the hidden number: 5.
Evaluation: Correct

# Example 2
Ground Truth Answer: animal
Model Response: The hidden word in the image is "ANIMAL".
Evaluation: Correct

# Example 3
Ground Truth Answer: A
Model Response: The hidden letter in the image is B.
Evaluation: Incorrect

# Example 4
Ground Truth Answer: 我
Model Response: The hidden Chinese character in the image is 我.
Evaluation: Correct

# Example 5
Ground Truth Answer: 你好吗
Model Response: The hidden Chinese characters in the image are 我好嘛.
Evaluation: Incorrect

Now it's your turn to evaluate.
Ground Truth Answer: [GROUND_TRUTH]
Model Response: [RESPONSE]
Evaluation:
"""