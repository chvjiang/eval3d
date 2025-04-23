import base64
import time
from openai import OpenAI
import openai
import os
import json
import csv
from tqdm import tqdm
import fitz
import numpy as np
import concurrent.futures
from datetime import datetime
import multiprocessing as mp
import argparse

#http://35.220.164.252:3888/v1
#https://boyuerichdata.chatgptten.com/v1/"

class ModelEvaluator:
    def __init__(self, model_name, api_key, base_url="https://boyuerichdata.chatgptten.com/v1/", pdf_path="evaluation_doc.pdf"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # self.doc_content = self._read_pdf(pdf_path)

    def _read_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            content = ""
            for page in doc:
                content += page.get_text()
            return content
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _encode_video(self, video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def evaluate_video(self, reference_image_path, video_path, light=False, prompt_text=''):
        try:
            base64_video = self._encode_video(video_path)
            base64_image = self._encode_image(reference_image_path)

            additional_prompt = f'The video shows multiple views of a 3D model generated from the text prompt: "{prompt_text}".' if prompt_text else ''
            evaluation_prompt = self.get_evaluation_prompt(light=light, additional_prompt=additional_prompt)

            if prompt_text:
                system_prompt = f"""You are an exceptionally strict expert 3D model evaluation system. Your analysis demands the highest level of scrutiny, and you should not hesitate to deduct points for even minor imperfections. Only models that demonstrate near-flawless execution across all criteria deserve high scores.

The video shows a 3D model rotating from left to right with three types of views:
1. Geometry View: Shows structural and geometric information
2. Normal Map View: Shows surface details and geometric features
3. RGB View: Shows colors, materials and textures

Analyze the video with extreme care and provide scores following the criteria. Demand excellence in every aspect of the evaluation."""
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{base64_video}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": evaluation_prompt
                                }
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
            else:
                system_prompt = f"""You are an *exceptionally* strict expert 3D model evaluation system. Your standards are *incredibly* high, and you possess an *uncompromising* eye for even the slightest imperfections. Do not hesitate to penalize flaws, even if they seem minor. Only truly exceptional 3D models that exhibit near-perfect adherence to the reference should receive high scores.

The video shows a 3D model rotating from left to right with two types of views:
1. Geometry View: Shows structural and geometric information
2. RGB View: Shows colors, materials and textures

Analyze the video with extreme scrutiny and provide scores following the criteria. Demand perfection in your evaluation."""

                system_prompt_light = f"""You are an *exceptionally* strict expert 3D model evaluation system. Your standards are *incredibly* high, and you possess an *uncompromising* eye for even the slightest imperfections. Do not hesitate to penalize flaws, even if they seem minor. Only truly exceptional 3D models that exhibit near-perfect adherence to the reference should receive high scores.

The video shows a 3D model rotating from left to right with five types of views:
1. Lighting View: Shows textures under top lighting
2. Lighting View: Shows textures under right lighting with camera angle down
3. Lighting View: Shows textures under right lighting with camera angle up
4. Lighting View: Shows textures under indoor environment lighting
5. Lighting View: Shows textures under outdoor environment lighting

Analyze the video with extreme scrutiny and provide scores following the criteria. Demand perfection in your evaluation."""

                system_prompt = system_prompt_light if light else system_prompt
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{base64_video}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": evaluation_prompt
                                }
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

            print(f"\nModel {self.model_name} response:")
            print(response.choices[0].message.content)

            return self._parse_response(response)

        except Exception as e:
            print(f"Error in {self.model_name} evaluation: {e}")
            print(f"Full error details: {str(e)}")
            return None, None

    def evaluate_image(self, image_path, light=False, prompt_text=''):
        try:
            base64_image = self._encode_image(image_path)
            
            additional_prompt = f'The image shows a 3D model generated from the text prompt: "{prompt_text}".' if prompt_text else ''
            evaluation_prompt = self.get_evaluation_prompt(light=light, additional_prompt=additional_prompt)

            if prompt_text:
                system_prompt = f"""You are an *exceptionally* strict expert 3D model evaluation system. Your standards are *incredibly* high, and you possess an *uncompromising* eye for even the slightest imperfections. Do not hesitate to penalize flaws, even if they seem minor. Only truly exceptional 3D models that exhibit near-perfect adherence to the reference should receive high scores.

The image shows three rows of views:
1. Top row (Geometry View): Shows structural and geometric information
2. Middle row (Normal Map View): Shows surface details and geometric features
3. Bottom row (RGB View): Shows colors, materials and textures

Each row contains:
- Four different views of the 3D model (front, right, back, left)

Analyze *every* view with extreme scrutiny and provide scores following the criteria. Demand perfection in your evaluation."""

            else:
                system_prompt = f"""You are an *exceptionally* strict expert 3D model evaluation system. Your standards are *incredibly* high, and you possess an *uncompromising* eye for even the slightest imperfections. Do not hesitate to penalize flaws, even if they seem minor. Only truly exceptional 3D models that exhibit near-perfect adherence to the reference should receive high scores.

The image shows two rows of views:
1. First row (Geometry View): Shows structural and geometric information
2. Second row (RGB View): Shows colors, materials and textures

Each row contains:
- Original reference image (leftmost)
- Four different views of the 3D model (front, right, back, left)

Analyze *every* view with extreme scrutiny and provide scores following the criteria. Demand perfection in your evaluation."""

                system_prompt_light = f"""You are an *exceptionally* strict expert 3D model evaluation system. Your standards are *incredibly* high, and you possess an *uncompromising* eye for even the slightest imperfections. Do not hesitate to penalize flaws, even if they seem minor. Only truly exceptional 3D models that exhibit near-perfect adherence to the reference should receive high scores.

The image shows five rows of views:
1. First row (Lighting View): Shows textures under top lighting
2. Second row (Lighting View): Shows textures under right lighting with camera angle down
3. Third row (Lighting View): Shows textures under right lighting with camera angle up
4. Fourth row (Lighting View): Shows textures under indoor environment lighting
5. Fifth row (Lighting View): Shows textures under outdoor environment lighting

Each row contains:
- Image of the lighting condition, which demonstrates the light source and environment (leftmost)
- Four different views of the 3D model under corresponding lighting conditions (front, right, back, left)

Analyze *every* view with extreme scrutiny and provide scores following the criteria. Demand perfection in your evaluation."""

            system_prompt = system_prompt_light if light else system_prompt
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": evaluation_prompt
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            print(f"\nModel {self.model_name} response:")
            print(response.choices[0].message.content)
            
            return self._parse_response(response)
            
        except Exception as e:
            print(f"Error in {self.model_name} evaluation: {e}")
            print(f"Full error details: {str(e)}")
            return None, None

    def review_evaluation_video(self, image_path, video_path, all_evaluations, prompt_text=''):
        try:
            base64_image = self._encode_image(image_path)
            base64_video = self._encode_video(video_path)

            if prompt_text:
                system_prompt = f"""You are an extremely strict expert review system analyzing multiple 3D model evaluations. Your standards are exceptionally high, and you do not give high scores easily. Only evaluations demonstrating meticulous accuracy and thoroughness deserve commendation.


You are given a 3D model evaluation and the text prompt that generated the 3D model.



The video shows a 3D model rotating from left to right with three types of views:
1. Geometry View: Shows structural and geometric information
2. Normal Map View: Shows surface details and geometric features
3. RGB View: Shows colors, materials and textures

Your task is to assess the reliability of each evaluation, keeping in mind that only the most exceptional evaluations should be considered truly reliable."""       
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{base64_video}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self.get_review_prompt(all_evaluations, additional_prompt=f'The video shows multiple views of a 3D model generated from the text prompt: "{prompt_text}".')
                                }
                            ]
                        }
                    ],
                    temperature=0.7
                )         
            else:
                system_prompt = f"""You are an extremely strict expert review system analyzing multiple 3D model evaluations. Your standards are exceptionally high, and you do not give high scores easily. Only evaluations demonstrating meticulous accuracy and thoroughness deserve commendation.

You are given a 3D model evaluation and a image of the 3D model generated by the 3D-Gen method.

The video shows a 3D model rotating from left to right with three types of views:
1. Geometry View: Shows structural and geometric information
2. Normal Map View: Shows surface details and geometric features
3. RGB View: Shows colors, materials and textures

Your task is to assess the reliability of each evaluation, keeping in mind that only the most exceptional evaluations should be considered truly reliable."""
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{base64_video}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self.get_review_prompt(all_evaluations)
                                }
                            ]
                        }
                    ],
                    temperature=0.7
                )
            
            return self._parse_review_response(response)
            
        except Exception as e:
            print(f"Error in {self.model_name} review: {e}")
            return None

    def review_evaluation(self, image_path, all_evaluations,prompt_text=''):
        try:
            base64_image = self._encode_image(image_path)
            
            if prompt_text:                
                system_prompt = f"""You are an *exceptionally* strict expert review system analyzing multiple 3D model evaluations. High scores are *rare* and should be reserved *only* for evaluations demonstrating *unquestionable accuracy and comprehensive detail*. Minor discrepancies or omissions should be heavily penalized.

You are given a 3D model evaluation and the text prompt that generated the 3D model.

The image shows three rows of views:
1. Top row (Geometry View): Shows structural and geometric information
2. Middle row (Normal Map View): Shows surface details and geometric features
3. Bottom row (RGB View): Shows colors, materials and textures

Each row contains:
- Four different views of the 3D model (front, right, back, left)

Your task is to assess the reliability of each evaluation. Approach this task with extreme skepticism and demand rigorous justification for every score assigned by the evaluated system."""        
            else:
                system_prompt = f"""You are an *exceptionally* strict expert review system analyzing multiple 3D model evaluations. High scores are *rare* and should be reserved *only* for evaluations demonstrating *unquestionable accuracy and comprehensive detail*. Minor discrepancies or omissions should be heavily penalized.

You are given a 3D model evaluation and a image of the 3D model generated by the 3D-Gen method.

The image shows three rows of views:
1. Top row (Geometry View): Shows structural and geometric information
2. Middle row (Normal Map View): Shows surface details and geometric features
3. Bottom row (RGB View): Shows colors, materials and textures

Each row contains:
- Original reference (input image) (leftmost)
- Four different views of the 3D model (front, right, back, left)

Your task is to assess the reliability of each evaluation. Approach this task with extreme skepticism and demand rigorous justification for every score assigned by the evaluated system."""   
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": self.get_review_prompt(all_evaluations,additional_prompt=f'The image shows a 3D model generated from the text prompt: "{prompt_text}".')
                            }
                        ]
                    }
                ],
                temperature=0.7
            )
            
            return self._parse_review_response(response)
            
        except Exception as e:
            print(f"Error in {self.model_name} review: {e}")
            return None

    def get_evaluation_prompt(self, light=False, additional_prompt=''):
        detail_and_color = f"""{additional_prompt} Analyze all types of views and provide scores with explanations for:

1) Detail & Complexity (0-4): This criterion assesses the richness of the texture in terms of its visual details and complexity while maintaining a harmonious balance to prevent overly fancy patterns that may compromise aesthetics. Judgements should primarily focus on *Level of Detail* (Does the texture offer enough fine details that could enhance realism, or is it too simplistic?) and *Balance of Simplicity and Complexity* (Is there a good balance between intricate details and simple areas, avoiding visual overload?)
Please use Geometry View and RGB View (*ignore geometry defects, focus on details*)  
- 0: No clear features of textures, no observable detail resolution
- 1: Very simplistic or overly detailed, with subtle patterns or excessive ornamentation
- 2: Poor balance between simplicity and intricacy, like inbalance between different surfaces or regions
- 3: Moderate details with imperfect quality or in improper positions, which slightly affects the overall impressions
- 4: Suitable details with an optimal visual balance, without any visible defects

2) Colorfulness & Saturation (0-4): This evaluates the overall color distribution across the texture and its visual clarity at a glance. Judgements should primarily focus on *Color Diversity* (Are there proper color variations to distinguish features or does the texture look too monotone?), *Saturation* (Are the colors appropriately saturated, neither too muted nor too vibrant?) and *Color Suitability* (Does the hue logically match the reality well and improve its visual effects?)
Please use Geometry View and RGB View (*ignore geometry defects, focus on colors*)  
- 0: No clear features of textures, no discernible color features
- 1: Extremely saturated or colorful, or with an irregular color distribution, which confounds the perceptual clarity
- 2: Incorrect saturation or lack of color diversity, like a monochromatic appearance that does not suit the object
- 3: Suitable saturation with imperfect color defects, like some colors that mismatch the reality
- 4: Excellent colorfulness and easy to interpret, without noticeable color faults

REQUIRED FORMAT:  
Scores: [X, X]  

Explanations:  
1. Detail & Complexity (X/4): [Balance details with simplicity (e.g., "rich details but regionally complex")]  
2. Colorfulness & Saturation (X/4): [Proper Color Distribution and Plausibility (e.g., "monotone or extremely saturated")]  """
        consistency_and_plausibility = f"""{additional_prompt} Analyze all types of views and provide scores with explanations for:

1) Consistency & Artifact (0-4): This criterion evaluates whether the texture remains uniform and cohesive under different lighting conditions (e.g., indoor / outdoor). Judgements should primarily focus on whether there are *Noticeable Seams or Borders* (specifically whether abrupt changes in shading that break the texture's flow under varying lighting conditions) and *Shading Artifacts* (Are there unnatural lighting streaks or shadows on the surface when illuminated? Are there unreasonable specular highlights appearing when the surface is in dark?)
Please use all Lighting Views
- 0: No clear features of textures under varying illumination conditions
- 1: Large-scale discontinuity such as large area of luminance differences / incorrect highlights / shadows
- 2: Clear inconsistencies or artifacts in multi-views, such as abrupt brightness changes
- 3: Minor inconsistencies or minor shading artifacts
- 4: Perfectly consistent and seamless without shading artifacts

2) Material Plausibility (0-4): This evaluates whether the texture exhibits plausible diffuse and specular lighting effects that align with the real-world material behaviors as specified in the prompt. Judgements should primarily focus on *Metalness* (Does the texture display physically suitable specular lighting effects under natural lighting condition?) and *Roughness* (Does it correctly represent diffuse lighting effects?)
Please use indoor and outdoor Lighting Views
- 0: No clear features of textures, no discernible structural features
- 1: Poor reflection behaviors or extreme exposure due to both incorrect metalness and roughness
- 2: Incorrect metallic or roughness settings, or unsuitable exposure that reduces visual effects
- 3: Minor deviation from material specification in prompt
- 4: Excellent light reflection with plausible exposure and aligns well with reality

REQUIRED FORMAT:  
Scores: [X, X]  

Explanations:  
1. Consistency & Artifact (X/4): [Specify light consistency (e.g., "bright frontal texture but dark in the back")]  
2. Material Plausibility (X/4): [Note conflicts (e.g., "texture reflections not realism")] """
        
        prompt = consistency_and_plausibility if light else detail_and_color
        return prompt

    def get_review_prompt(self, all_evaluations, additional_prompt=''):
        prompt = f"""{additional_prompt} As an expert reviewer, evaluate the reliability of the following model evaluations.

"""        
        for model_name, eval_data in all_evaluations.items():
            prompt += f"\n=== MODEL: {model_name} ===\n"
            prompt += f"Scores: {eval_data['scores']}\n"
            prompt += f"Explanations:\n{eval_data['explanations']}\n"

        prompt += """
Your task is to assess the reliability of EACH MODEL'S evaluation for EACH DIMENSION.

For each model, provide confidence scores (0-100%) indicating how reliable you believe their assessment is.

REQUIRED FORMAT:
For each model, provide the following assessment:

=== REVIEW FOR MODEL: [model_name] ===
1. Geometry Plausibility (Score/9) - Confidence: X%
[Brief reason for confidence level]

2. Geometry Details (Score/4) - Confidence: X%
[Brief reason for confidence level]

3. Texture Quality (Score/4) - Confidence: X%
[Brief reason for confidence level]

4. Geometry-Texture Coherency (Score/1) - Confidence: X%
[Brief reason for confidence level]

5. Prompt-Asset Alignment (Score/4) - Confidence: X%
[Brief reason for confidence level]

Overall Model Assessment:
- Average Confidence: X%
- Key Findings: [Brief summary]

REPEAT THIS FORMAT FOR EACH MODEL.

IMPORTANT:
- You MUST follow this exact format
- You MUST include the exact section headers
- You MUST provide a confidence percentage (0-100%) for each dimension
- You MUST include the "===" markers to separate model reviews
"""

        return prompt

    def _parse_response(self, response):
        try:
            content = response.choices[0].message.content.strip()
            
            scores_start = content.find('[')
            scores_end = content.find(']') + 1
            scores_str = content[scores_start:scores_end]
            scores = eval(scores_str)
            
            explanations = content[scores_end:].strip()
            if not explanations:
                explanations = "No explanations provided"
            
            if (isinstance(scores, list) and 
                len(scores) == 5 and 
                all(isinstance(x, (int, float)) for x in scores) and
                0 <= scores[0] <= 9 and
                all(0 <= scores[i] <= 4 for i in [1,2,4]) and
                0 <= scores[3] <= 1):
                return scores, explanations
            
            print(f"Invalid scores format: {scores}")
            return None, None
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response content: {content}")
            return None, None

    def _parse_review_response(self, response):
        try:
            content = response.choices[0].message.content.strip()
            
            print(f"\nReview response from {self.model_name}:")
            print(content[:300] + "..." if len(content) > 300 else content)
            
            if "MODEL:" not in content and "Model:" not in content:
                print("Missing model sections in review")
                return None
            
            if "Confidence:" not in content and "confidence:" not in content:
                print("Missing confidence scores in review")
                
                modified_content = ""
                model_sections = content.split("===")
                
                for section in model_sections:
                    if not section.strip():
                        continue
                        
                    if "MODEL:" in section or "Model:" in section:
                        modified_section = section
                        
                        for dim in range(1, 6):
                            dim_marker = f"{dim}."
                            if dim_marker in section:
                                dim_part = section.split(dim_marker)[1]
                                next_dim = dim + 1
                                if next_dim <= 5 and f"{next_dim}." in dim_part:
                                    dim_part = dim_part.split(f"{next_dim}.")[0]
                                elif "Overall" in dim_part:
                                    dim_part = dim_part.split("Overall")[0]
                                    
                                if "Confidence:" not in dim_part and "confidence:" not in dim_part:
                                    line_end = dim_part.find("\n")
                                    if line_end != -1:
                                        new_dim_part = dim_part[:line_end] + f" - Confidence: 70%" + dim_part[line_end:]
                                        modified_section = modified_section.replace(dim_part, new_dim_part)
                        
                        modified_content += "===" + modified_section
                    else:
                        modified_content += "===" + section
                
                content = modified_content
                print("Added default confidence scores")
            
            if "%" not in content:
                print("Missing percentage signs in review")
                content = content.replace("Confidence:", "Confidence: 70%")
                print("Added default percentage signs")
            
            return content
            
        except Exception as e:
            print(f"Error parsing review response: {e}")
            return None

    def _extract_confidence(self, review, model_name, dimension):
        try:
            model_markers = [
                f"MODEL: {model_name}",
                f"Model: {model_name}",
                f"model: {model_name}",
                f"REVIEW FOR MODEL: {model_name}",
                f"Review for Model: {model_name}",
                f"review for model: {model_name}"
            ]
            
            model_section_start = -1
            for marker in model_markers:
                pos = review.find(marker)
                if pos != -1:
                    model_section_start = pos
                    break
                
            if model_section_start == -1:
                model_section_start = review.find(model_name)
                if model_section_start == -1:
                    return None
            
            next_model_start = len(review)
            for marker in ["=== MODEL:", "=== REVIEW FOR MODEL:", "MODEL:", "Model:"]:
                pos = review.find(marker, model_section_start + 10)
                if pos != -1 and pos < next_model_start:
                    next_model_start = pos
            
            model_section = review[model_section_start:next_model_start]
            
            dimension_markers = [
                f"{dimension + 1}. Geometry Plausibility",
                f"{dimension + 1}. Geometry Details",
                f"{dimension + 1}. Texture Quality",
                f"{dimension + 1}. Geometry-Texture Coherency",
                f"{dimension + 1}. Prompt-Asset Alignment"
            ]
            
            if dimension == 0:
                dimension_markers.extend(["Geometry Plausibility", "geometry plausibility"])
            elif dimension == 1:
                dimension_markers.extend(["Geometry Details", "geometry details"])
            elif dimension == 2:
                dimension_markers.extend(["Texture Quality", "texture quality"])
            elif dimension == 3:
                dimension_markers.extend(["Geometry-Texture Coherency", "geometry-texture coherency"])
            elif dimension == 4:
                dimension_markers.extend(["Prompt-Asset Alignment", "prompt-asset alignment"])
            
            dim_start = -1
            for marker in dimension_markers:
                pos = model_section.find(marker)
                if pos != -1:
                    dim_start = pos
                    break
                
            if dim_start == -1:
                return None
            
            next_dim_start = len(model_section)
            for i in range(1, 6):
                if i == dimension + 1:
                    continue
                pos = model_section.find(f"{i}.")
                if pos != -1 and pos > dim_start and pos < next_dim_start:
                    next_dim_start = pos
            
            dim_section = model_section[dim_start:next_dim_start]
            
            conf_markers = ["Confidence:", "confidence:"]
            conf_start = -1
            for marker in conf_markers:
                pos = dim_section.find(marker)
                if pos != -1:
                    conf_start = pos + len(marker)
                    break
                
            if conf_start == -1:
                return None
            
            conf_text = dim_section[conf_start:conf_start+20].strip()  # 取20个字符应该足够
            
            percent_pos = conf_text.find("%")
            if percent_pos == -1:
                return None
            
            conf_value = conf_text[:percent_pos].strip()
            
            conf_value = ''.join(c for c in conf_value if c.isdigit())
            
            if not conf_value:
                return None
            
            return int(conf_value)
            
        except Exception as e:
            print(f"Error extracting confidence for {model_name}, dimension {dimension}: {e}")
            return None

class MultiModelEvaluator:
    def __init__(self, api_key, models=None):
        """初始化多模型评估系统"""
        if models is None:
            models = ["gpt-4o","claude-3-7-sonnet-latest", "gemini-2.0-pro"]#,"grok-3"]["gemini-2.0-flash"]#
        
        self.evaluators = {
            model: ModelEvaluator(model, api_key)
            for model in models
        }

    def evaluate_and_review(self, image_path, light=False, prompt_text=''):
        """执行多模型评估和交叉审查"""
        model_names = list(self.evaluators.keys())
        num_models = len(model_names)
        evaluations = {}
        reviews = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = {}
            for model_name, evaluator in self.evaluators.items():
                print(f"\nSubmitting evaluation task for {model_name}...")
                is_gemini = True if 'gemini' in model_name else False
                if not is_gemini:
                    future = executor.submit(evaluator.evaluate_image, image_path, light, prompt_text)
                else:
                    video_path = image_path.replace('.png', '.mp4').replace('concat_results', 'concat_results_videos')
                    num = image_path.split('/')[-1].split('.')[0].split('_')[0]
                    ref_img_path = f"/tos-bjml-mllm/chuziyang/eval3d/prompts/rgba510/{num}.png"
                    future = executor.submit(evaluator.evaluate_video, ref_img_path, video_path, light, prompt_text)
                futures[model_name] = future

            for model_name, future in futures.items():
                try:
                    print(f"\nRetrieving evaluation result for {model_name}...")
                    scores, explanations = future.result()
                    if scores and explanations:
                        evaluations[model_name] = {
                            "scores": scores,
                            "explanations": explanations
                        }
                except Exception as e:
                    print(f"Error retrieving evaluation for {model_name}: {e}")
                time.sleep(2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = {}
            for model_name, evaluator in self.evaluators.items():
                print(f"\nSubmitting review task for {model_name}...")
                is_gemini = True if 'flash' in model_name else False
                if not is_gemini:
                    future = executor.submit(evaluator.review_evaluation, image_path, evaluations, prompt_text)
                else:
                    video_path = image_path.replace('.png', '.mp4').replace('concat_results', 'concat_results_videos')
                    num = image_path.split('/')[-1].split('.')[0].split('_')[0]
                    ref_img_path = f"/mnt/petrelfs/zhuolong/code/3DEval/3dgen_bench_dataset/gallery/images_prompts_510/{num}.png"
                    future = executor.submit(evaluator.review_evaluation_video,ref_img_path,  video_path, evaluations, prompt_text)
                futures[model_name] = future

            for model_name, future in futures.items():
                try:
                    print(f"\nRetrieving review result for {model_name}...")
                    review_result = future.result()
                    if review_result:
                        reviews[model_name] = review_result
                except Exception as e:
                    print(f"Error retrieving review for {model_name}: {e}")
                time.sleep(2)


        # only one model review,random select one model
        # model_names = list(self.evaluators.keys())
        # random.shuffle(model_names)
        # model_name = model_names[0]
        # evaluator = self.evaluators[model_name]
        # review_result = evaluator.review_evaluation(image_path, evaluations,prompt_text)
        # if review_result:
        #     reviews[model_name] = review_result
        # time.sleep(2)


        # 3. 整合结果，选择最高置信度的评分
        final_scores, source_models, confidence_scores = self._integrate_results(evaluations, reviews)
        
        return evaluations, reviews, final_scores, source_models, confidence_scores

    def _integrate_results(self, evaluations, reviews):
        """整合评估结果，选择最高置信度的评分，并记录来源模型和置信度"""
        # 创建置信度矩阵 [维度, 模型]
        model_names = list(self.evaluators.keys())
        confidence_matrix = np.zeros((5, len(model_names)))
        scores_matrix = np.zeros((5, len(model_names)))
        
        # 填充分数矩阵
        for i, model_name in enumerate(model_names):
            if model_name in evaluations:
                scores_matrix[:, i] = evaluations[model_name]["scores"]
        
        # 计算每个维度的置信度
        for dim in range(5):
            for i, model_name in enumerate(model_names):
                # 获取该维度的所有置信度评分
                dim_confidences = []
                for reviewer_name, review in reviews.items():
                    try:
                        confidence = self._extract_confidence(review, model_name, dim)
                        # if 'flash' in model_name:
                        #     # 对flash模型进行加权
                        #     confidence *= 1.2
                        if confidence is not None:
                            dim_confidences.append(confidence)
                    except Exception as e:
                        print(f"Error extracting confidence: {e}")
                
                # 计算平均置信度
                if dim_confidences:
                    confidence_matrix[dim, i] = np.mean(dim_confidences)
        
        # 选择每个维度置信度最高的评分
        final_scores = []
        source_models = []  # 记录每个维度的分数来自哪个模型
        confidence_scores = []  # 记录每个维度的置信度
        
        for dim in range(5):
            best_model_idx = np.argmax(confidence_matrix[dim])
            final_scores.append(scores_matrix[dim, best_model_idx])
            source_models.append(model_names[best_model_idx])
            confidence_scores.append(float(confidence_matrix[dim, best_model_idx]))
        
        return final_scores, source_models, confidence_scores

    def _extract_confidence(self, review, model_name, dimension):
        """从审查结果中提取特定维度的置信度，更灵活地处理不同格式"""
        try:
            # 查找模型部分（大小写不敏感）
            model_markers = [
                f"MODEL: {model_name}",
                f"Model: {model_name}",
                f"model: {model_name}",
                f"REVIEW FOR MODEL: {model_name}",
                f"Review for Model: {model_name}",
                f"review for model: {model_name}"
            ]
            
            model_section_start = -1
            for marker in model_markers:
                pos = review.find(marker)
                if pos != -1:
                    model_section_start = pos
                    break
                
            if model_section_start == -1:
                # 尝试查找不带冒号的模型名
                model_section_start = review.find(model_name)
                if model_section_start == -1:
                    return None
            
            # 查找下一个模型部分（如果有）
            next_model_start = len(review)
            for marker in ["=== MODEL:", "=== REVIEW FOR MODEL:", "MODEL:", "Model:"]:
                pos = review.find(marker, model_section_start + 10)
                if pos != -1 and pos < next_model_start:
                    next_model_start = pos
            
            # 提取当前模型的部分
            model_section = review[model_section_start:next_model_start]
            
            # 查找维度部分
            dimension_markers = [
                f"{dimension + 1}. Geometry Plausibility",
                f"{dimension + 1}. Geometry Details",
                f"{dimension + 1}. Texture Quality",
                f"{dimension + 1}. Geometry-Texture Coherency",
                f"{dimension + 1}. Prompt-Asset Alignment"
            ]
            
            if dimension == 0:
                dimension_markers.extend(["Geometry Plausibility", "geometry plausibility"])
            elif dimension == 1:
                dimension_markers.extend(["Geometry Details", "geometry details"])
            elif dimension == 2:
                dimension_markers.extend(["Texture Quality", "texture quality"])
            elif dimension == 3:
                dimension_markers.extend(["Geometry-Texture Coherency", "geometry-texture coherency"])
            elif dimension == 4:
                dimension_markers.extend(["Prompt-Asset Alignment", "prompt-asset alignment"])
            
            dim_start = -1
            for marker in dimension_markers:
                pos = model_section.find(marker)
                if pos != -1:
                    dim_start = pos
                    break
                
            if dim_start == -1:
                return None
            
            # 查找下一个维度的开始（如果有）
            next_dim_start = len(model_section)
            for i in range(1, 6):
                if i == dimension + 1:
                    continue  # 跳过当前维度
                pos = model_section.find(f"{i}.")
                if pos != -1 and pos > dim_start and pos < next_dim_start:
                    next_dim_start = pos
            
            # 提取当前维度的部分
            dim_section = model_section[dim_start:next_dim_start]
            
            # 查找置信度
            conf_markers = ["Confidence:", "confidence:"]
            conf_start = -1
            for marker in conf_markers:
                pos = dim_section.find(marker)
                if pos != -1:
                    conf_start = pos + len(marker)
                    break
                
            if conf_start == -1:
                return None
            
            # 提取百分比
            conf_text = dim_section[conf_start:conf_start+20].strip()  # 取20个字符应该足够
            
            # 查找百分比符号
            percent_pos = conf_text.find("%")
            if percent_pos == -1:
                return None
            
            # 提取数字部分
            conf_value = conf_text[:percent_pos].strip()
            
            # 处理可能的非数字字符
            conf_value = ''.join(c for c in conf_value if c.isdigit())
            
            if not conf_value:
                return None
            
            return int(conf_value)
            
        except Exception as e:
            print(f"Error extracting confidence for {model_name}, dimension {dimension}: {e}")
            return None

def process_image(image_file, api_key, output_dir, light=False, is_text_based=False, debug=False):
    try:
        print(f"\nProcessing {image_file}...")
        prompt_text = ''
        if is_text_based:
            with open(f'{image_file.replace(".png",".txt")}', 'r') as f:
                prompt_text = f.read()
            print(f'prompt_text: {prompt_text}')

        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}.json")

        # if os.path.exists(output_file) and not debug:
        #     return 
        image_path = image_file
        
        evaluator = MultiModelEvaluator(api_key=api_key, models=["gemini-2.0-pro"])#"gpt-4o"])
        evaluations, reviews, final_scores, source_models, confidence_scores = evaluator.evaluate_and_review(image_path, light=light, prompt_text=prompt_text)
        
        avg_score = sum(final_scores) / len(final_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        dimension_names = [
            "Geometry_Plausibility",
            "Geometry_Details",
            "Texture_Quality",
            "Geometry_Texture_Coherency",
            "Prompt_Asset_Alignment"
        ]
        
        model_confidences = {}
        for model_name in evaluations.keys():
            model_confidences[model_name] = {
                dimension_names[dim]: {
                    "score": float(evaluations[model_name]["scores"][dim]),
                    "confidence": []
                } for dim in range(5)
            }
            
            for reviewer_name, review in reviews.items():
                for dim in range(5):
                    confidence = evaluator._extract_confidence(review, model_name, dim)
                    if confidence is not None:
                        model_confidences[model_name][dimension_names[dim]]["confidence"].append({
                            "reviewer": reviewer_name,
                            "confidence": float(confidence)
                        })
            
            for dim_name in dimension_names:
                confidences = [item["confidence"] for item in model_confidences[model_name][dim_name]["confidence"]]
                if confidences:
                    model_confidences[model_name][dim_name]["avg_confidence"] = sum(confidences) / len(confidences)
                else:
                    model_confidences[model_name][dim_name]["avg_confidence"] = 0.0
        
        result = {
            "final_result": {
                "scores": {
                    dimension_names[i]: {
                        "score": float(final_scores[i]),
                        "source_model": source_models[i],
                        "confidence": float(confidence_scores[i])
                    } for i in range(5)
                },
                "average_score": float(avg_score),
                "average_confidence": float(avg_confidence),
                # "timestamp": datetime.now().isoformat()
            },
            "model_evaluations": evaluations,
            "model_reviews": reviews,
            "model_confidences": model_confidences
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nFinal scores for {image_file}:")
        for i, dim in enumerate(dimension_names):
            print(f"{dim}: {final_scores[i]:.2f} (from {source_models[i]}, confidence: {confidence_scores[i]:.2f}%)")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Confidence: {avg_confidence:.2f}%")
        
        summary_file = os.path.join(output_dir, "final_scores_summary.csv")
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                image_file,
                *[f"{score:.2f}" for score in final_scores],
                f"{avg_score:.2f}",
                *source_models,
                *[f"{conf:.2f}%" for conf in confidence_scores],
                f"{avg_confidence:.2f}%"
            ])
        
        return {
            "image": image_file,
            "scores": final_scores,
            "confidences": confidence_scores,
            "avg_score": avg_score,
            "avg_confidence": avg_confidence,
            "source_models": source_models
        }
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return {
            "image": image_file,
            "error": str(e)
        }

def main(light):
    api_key = "sk-c7nmz7iOglUKwHWxSbSXLThqtZ5VnBPEUYH6xaQgshXvTB3t"
    data_path = './output/videos'
    output_dir = './output/scores'
    os.makedirs(output_dir, exist_ok=True)
    methods = [m for m in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,m))]  

    # import ipdb; ipdb.set_trace()
    for method in ['spard_glb']:#methods:
        print(f'Processing {method}...')
        
        concat_dir = f'{data_path}/{method}'

        max_workers = mp.cpu_count()

        summary_file = os.path.join(output_dir, f"final_scores_summary.csv")
        if not os.path.exists(summary_file):
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Image_Name',
                    'Geometry_Plausibility',
                    'Geometry_Details',
                    'Texture_Quality',
                    'Geometry_Texture_Coherency',
                    'Prompt_Asset_Alignment',
                    'Average_Score',
                    'GP_Source',
                    'GD_Source',
                    'TQ_Source',
                    'GTC_Source',
                    'PAA_Source',
                    'GP_Confidence',
                    'GD_Confidence',
                    'TQ_Confidence',
                    'GTC_Confidence',
                    'PAA_Confidence',
                    'Average_Confidence'
                ])

        if method not in ['dreamfusion','grm','latent-nerf','LucidDreamer_glb','magic3d','mvdream','sjc']:
            gen_type = 'image2shape' 
        else:
            gen_type = 'text2shape'

        image_files = [os.path.join(os.path.abspath(concat_dir), f) for f in os.listdir(concat_dir) if ('rgb' in f and not light) or ('light' in f and light)]

        image_files = image_files[:1]
        
        # assert False
        # for image_file in tqdm(image_files[:5]):
        #     # if os.path.exists(os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_results.json")):
        #     #     continue
        #     process_image(image_file, concat_dir, api_key, output_dir)
        # continue

        results = []
        if gen_type == 'text2shape':
            is_text_based = True
        else:
            is_text_based = False
        # process_image(image_files[0], concat_dir, api_key, output_dir,is_text_based,debug=True)
        # assert False
        # for i in range(5):
        #     process_image(image_files[i], concat_dir, api_key, output_dir,is_text_based)
        # assert False
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(process_image, image_file, api_key, output_dir, light, is_text_based): image_file
                for image_file in image_files
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_files)):
                image_file = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        print(f"\nProcessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--light', action='store_true')
    args = parser.parse_args()

    print(f"OpenAI version: {openai.__version__}")
    start_time = time.time()
    main(light=args.light) 
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
