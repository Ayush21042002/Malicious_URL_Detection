from flask import Flask, render_template, request
import tldextract
import pandas as pd
from urllib.parse import urlparse
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")
with open('utils/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label_dict = {}
for feature in ['subdomain', 'domain', 'domain_suffix']:
    with open(f'utils/label_{feature}_dictionary.pkl', 'rb') as f:
      label_dict[feature] = pickle.load(f)

def parsed_url(url):
    # extract subdomain, domain, and domain suffix from url
    # if item == '', fill with '<empty>'
    subdomain, domain, domain_suffix = (
        '<empty>' if extracted == '' else extracted for extracted in tldextract.extract(url))

    return [subdomain, domain, domain_suffix]


def extract_url(data):
    # parsed url
    extract_url_data = [parsed_url(url) for url in data['url']]
    extract_url_data = pd.DataFrame(extract_url_data, columns=[
                                    'subdomain', 'domain', 'domain_suffix'])

    # concat extracted feature with main data
    data = data.reset_index(drop=True)
    data = pd.concat([data, extract_url_data], axis=1)

    return data


def prediction(url):
  url = urlparse(url)
  url = url.netloc + url.path + url.params + url.query + url.fragment
  df = pd.DataFrame([url], columns=['url'])
  df = extract_url(df)
  df_seq = tokenizer.texts_to_sequences(df['url'])
  sequence_length = 161
  df_seq = pad_sequences(df_seq, padding='post', maxlen=sequence_length)
  for feature in ['subdomain', 'domain', 'domain_suffix']:
    label_index = label_dict[feature]
    df.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>']
                          for val in df.loc[:, feature]]
  df = [df_seq, df['subdomain'], df['domain'], df['domain_suffix']]
  return model.predict(df)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  url = request.get_json(force=True)['url']
  print(url)
  confidence = prediction(url)[0][0]
  print(confidence)
  if confidence >= 0.75:
    return "Danger"
  else:
    return "Safe"

if __name__ == "__main__":
    app.debug = True
    app.run()
