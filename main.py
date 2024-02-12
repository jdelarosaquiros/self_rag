from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import requests
import string
from lxml import html
from googlesearch import search
from bs4 import BeautifulSoup
import re
import json
import time

# to search
# print(chatbot_query('how old is samuel l jackson'))

def chatbot_query(query, index=0):
    fallback = 'Sorry, I cannot think of a reply for that.'
    result = ''

    try:
        search_result_list = list(search(query, tld="co.in", num=10, stop=3, pause=1))
        print(search_result_list)

        # page = requests.get(search_result_list[index])
        page = requests.get("https://en.wikipedia.org/wiki/Computer_science")

        tree = html.fromstring(page.content)

        soup = BeautifulSoup(page.content, features="lxml")
        # print(soup)

        article_text = ''
        article = soup.findAll('p')
        for element in article:
            article_text += '\n' + ''.join(element.findAll(string = True))
        article_text = article_text.replace('\n', '')
        first_sentence = article_text.split('.')
        first_sentence = first_sentence[0].split('?')[0]

        chars_without_whitespace = first_sentence.translate(
            { ord(c): None for c in string.whitespace }
        )

        if len(chars_without_whitespace) > 0:
            result = first_sentence
        else:
            result = fallback

        return result
    except:
        if len(result) == 0: result = fallback
        return result

# create a function to search the web, get the top result, and return the text content
# def get_top_search_result(query):
#   try:
#     search_results = list(search(query, tld="co.in", num=1, stop=1, pause=1))
#     # search_results = ['https://en.wikipedia.org/wiki/Computer_science']
#     if search_results:
#       page = requests.get(search_results[0])
#       soup = BeautifulSoup(page.content, features="lxml")
#       article_text = ''
#       article = soup.findAll('p')
#       for element in article:
#         article_text += '\n' + ''.join(element.findAll(string=True))
#       article_text = article_text.replace('\n', '')
#       return article_text
#     else:
#       return "No search results found."
#   except:
#     return "An error occurred while searching the web."

def get_top_search_result(query):
  try:
    fallback = 'Sorry, I cannot think of a reply for that.'
    search_results = list(search(query, tld="co.in", num=10, stop=5, pause=1))
    results: list[str] = []
    print(search_results)
    # search_results = ['https://en.wikipedia.org/wiki/Computer_science']
    if not search_results:
      return "No search results found."
    for result in search_results:
        # print(result)
        page = requests.get(result, timeout=2)
        print("here")
        if page.status_code != 200:
            continue
        soup = BeautifulSoup(page.content, features="lxml")
        article_text:str = ''
        article = soup.findAll('p')
        # print(article)
        for element in article:
            article_text += '\n' + ''.join(element.findAll(string=True))
        article_text = article_text.replace('\n', '')
        first_sentence = article_text.split('.')
        first_sentence = first_sentence[0].split('?')[0]

        chars_without_whitespace = first_sentence.translate(
            { ord(c): None for c in string.whitespace }
        )

        if len(chars_without_whitespace) > 0:
            result = article_text
        else:
            result = fallback

        print(result)
        results.append(result)
    return results
  except:
    print("An error occurred while searching the web.")
    return results

token_dict = {'Relevant': 1, 'Irrelevant': 0,
              'Fully supported': 10, 'Partially supported': 8, 'No support / Contradictory': 0,
              'Utility:5': 5, 'Utility:4': 4, 'Utility:3': 3, 'Utility:2': 2, 'Utility:1': 1,
              'No Retrieval': 0, 'Retrieval': 1}

relevant_tokens = {'Relevant': 1, 'Irrelevant': 0}
support_tokens = {'Fully supported': 10, 'Partially supported': 8, 'No support / Contradictory': 0}
utility_tokens = {'Utility:5': 5, 'Utility:4': 4, 'Utility:3': 3, 'Utility:2': 2, 'Utility:1': 1}
retrieval_tokens = {'No Retrieval': 0, 'Retrieval': 1}

class Prediction:
  def __init__(self, text, passage, pred_obj):
      
    tokens = re.findall(r'\[(.*?)\]', text)
    self.text = text
    self.passage = passage
    self.relevant_score = 0
    self.support_score = 0
    self.utility_score = 0
    self.pred_obj = pred_obj

    for token in tokens:
      if token in relevant_tokens:
        self.relevant_score = relevant_tokens[token]
      elif token in support_tokens:
        self.support_score = support_tokens[token]
      elif token in utility_tokens:
        self.utility_score = utility_tokens[token]
      

  def get_score(self):
    return self.relevant_score + self.support_score + self.utility_score
  
  def __str__(self):     
    return "Prediction: " + str({self.text, self.relevant_score, self.support_score, self.utility_score})


def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

def get_best_prediction_compose(preds, passages):
  best_prediction = Prediction(preds[0].outputs[0].text, 0, preds[0])
  for id, pred in enumerate(preds):
    prediction = Prediction(pred.outputs[0].text, passages[id], pred)

    if(best_prediction.get_score() < prediction.get_score()):
        best_prediction = prediction
  return best_prediction

def get_best_prediction(predictions: list[Prediction]):
  if(len(predictions) == 0): return None

  best_prediction = predictions[0]

  for prediction in predictions:
    if(best_prediction.get_score() < prediction.get_score()):
        best_prediction = prediction

  return best_prediction

# Start Script
model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
sampling_params = SamplingParams(temperature=0.4, top_p=1.0, max_tokens=100, skip_special_tokens=False, logprobs=2, prompt_logprobs=2)

start_time = time.time()
prompt = 'how many students are at utsa?'
results = get_top_search_result(prompt)
best_predictions = []

if(len(results ) < 1):
  print("No web page found")
  exit()


for result in results:
  result = r"Office of Institutional Research and Analysis.Overall, enrollment is up 5% from pre-pandemic figures, when 32,594 students were enrolled in fall 2019. First-time freshman enrollment also increased 1.8% to a record 5,600, as more college-bound students choose UTSA to pursue their bold futures.“UTSA has been successful in maintaining our enrollment gains during the pandemic thanks to strategic planning and a deliberate focus on increasing financial aid,” said Kimberly Andrews Espy, UTSA provost and senior vice president of academic affairs. Of particular note is a 41% surge in online enrollment from fall 2021. The university now has nearly 790 students enrolled in seven fully online degree and certificate programs. Eighty-five percent of online students are undergraduates. Other highlights: More than 1,170 students are entering UTSA as part of the university’s Bold Promise Program, the largest cohort to date. Raising the Bold Promise family income eligibility threshold to $70,000, made possible by UT System’s Promise Plus endowment, enabled more students to enroll through the Bold Promise Program. Additionally, UTSA is injecting an additional $2 million per year into financial aid from an endowment created from a $40 million gift to the university by philanthropist MacKenzie Scott.Nearly 77% of new Bold Promise students are Hispanic, and 39.8% are from Bexar County, reflecting the university’s commitment as an urban serving institution that provides high-quality, affordable education to students from the greater San Antonio community. More Bold Promise students are taking advantage of all UTSA has to offer by living on campus, as well. More than 46% of new Bold Promise students are living in on-campus housing, up from 22% in fall 2020. The increase is due in part to the university’s new Bold Scholar program, a comprehensive on-campus living-learning experience that is providing housing scholarships to 220 students.The university’s transfer student enrollment is 2,900 this fall, down from fall 2021 and consistent with declines in transfer student enrollment across the country following the COVID-19 pandemic.In addition to these enrollment figures, UTSA posted notable progress in several key student success metrics. In the last five years, degrees awarded has grown by 16% overall. Five-year growth in undergraduate degrees is 18%, from 4,938 in 2017-2018 to 5,834 in 2021-2022."
  # print(result)
  passages = [result[i:i+1500] for i in range(0, len(result), 1500)]
  print('Num Result Parts: ', len(passages))

  # prompts = [format_prompt(prompt, paragraph=passage) for passage in result_parts]

  preds = model.generate([format_prompt(prompt, paragraph=passage) for passage in passages], sampling_params)
  best_predictions.append(get_best_prediction_compose(preds, passages))

  # preds[0].outputs[0].logprobs.
  # preds[0].prompt_logprobs
  print("Prompt Probs: ", preds[0].prompt_logprobs)
  print("Output Probs: ", [pred.outputs[0].logprobs for pred in preds])
  break


  

if(len(best_predictions) <= 0):
  print("No predictions found")
  exit()

best_prediction = get_best_prediction(best_predictions)
final_results = {"prompt_tok": best_prediction.pred_obj.prompt_token_ids, "prompt_probs": best_prediction.pred_obj.prompt_logprobs,
                  "output_tok": best_prediction.pred_obj.outputs[0].token_ids, "output_probs": best_prediction.pred_obj.outputs[0].logprobs}

print("Output Probs: ", best_prediction.pred_obj.outputs[0].logprobs)
print("Prompt Probs: ", best_prediction.pred_obj.prompt_logprobs)

with open("test.json", "w") as outfile:
        json.dump(final_results, outfile)

end_time = time.time()
execution_time = end_time - start_time

print("Passage Cited:", passages[best_prediction.passage])
print(best_prediction.text)
print("Execution time:", execution_time, "seconds")
