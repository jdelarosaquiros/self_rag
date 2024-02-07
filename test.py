# from googlesearch import GoogleSearch
import requests
from googlesearch import search
from bs4 import BeautifulSoup
import re
import string

# response = GoogleSearch().search("what is computer science")
# for result in response.results:
#     print("Title: " + result.title)
#     print("Content: " + result.getText())

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
            result = first_sentence
        else:
            result = fallback

        print(result)
        results.append(result)
    return results
  except:
    print("An error occurred while searching the web.")
    return ["Sorry, an error occurred while searching the web."]
results = get_top_search_result("how many students are at utsa?")
print("Num Results", results)
# print(results[0])