import requests
from urllib.parse import urlparse, urljoin, parse_qs
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflowjs as tfjs

def get_url_features(url):
    features = {
        'url_length': 0,
        'has_special_chars': 0,
        'external_links_count': 0,
        'images_count': 0,
        'scripts_count': 0,
        'text_length': 0,
        'forms_count': 0,
        'login_forms_count': 0,
        'inline_scripts_count': 0,
        'iframes_count': 0,
        'suspicious_words_count': 0,
        'email_links_count': 0,
        'has_google_analytics': 0,
        'redirect_count': 0,
        'misspellings_in_title': 0,
        'suspicious_keywords_in_title': 0,
        'special_chars': 0
    }
    
    # URL Features
    features['url_length'] = len(url)
    features['has_special_chars'] = int(any(char in url for char in ['@', '-', '_', "!"]))
    
    # Fetch the webpage content
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        content = response.text
    except Exception as e:
        print(f"Error fetching the URL: {e}. ")
        return {"special_chars":0,"suspicious_keywords_in_title":0,'misspellings_in_title':0,'url_length': 0, 'has_special_chars': 0, 'external_links_count': 0, 'images_count': 0, 'scripts_count': 0, 'text_length': 0, 'forms_count': 0, 'login_forms_count': 0, 'inline_scripts_count': 0, 'iframes_count': 0, 'suspicious_words_count': 0, 'email_links_count': 0, 'has_google_analytics': 0, 'redirect_count': 0}
    # Extract the page title
    start_title_index = content.find('<title>')
    end_title_index = content.find('</title>')
    title = content[start_title_index:end_title_index]

    # Excessive Use of Special Characters
    if len(re.findall(r'[!$%^&*()<>?/\\|}{~:]', title)) > 3:
        features["special_chars"] = 1
 
    
    # Misspellings and Poor Grammar (basic example using a simple word list)
    common_words = ["the", "of", "and", "to", "a", "in", "is", "it", "you", "that"]
    words_in_title = title.split()
    if any(word not in common_words and not re.match(r'^[A-Za-z]+$', word) for word in words_in_title):
        features["misspellings_in_title"] = 1
    else:
        features["misspellings_in_title"] = 0
    # Suspicious Keywords
    suspicious_keywords = ["cheap", "deal", "cash", "earn", "prize", "gift", "free", "congratulations", "winner", "limited time offer"]
    if any(keyword in title.lower() for keyword in suspicious_keywords):
        features["suspicious_keywords_in_title"] = 1
    else:
        features["suspicious_keywords_in_title"] = 0

    #external links
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    external_links_count = 0
    start_a_index = 0
    while True:
        start_a_index = content.find('<a ', start_a_index)
        if start_a_index == -1:
            break
        end_a_index = content.find('>', start_a_index)
        a_tag = content[start_a_index:end_a_index]
        href_index = a_tag.find('href=')
        if href_index != -1:
            href_start = href_index + len('href=') + 1
            href_end = a_tag.find(a_tag[href_start-1], href_start)
            link = a_tag[href_start:href_end]
            link_domain = urlparse(urljoin(url, link)).netloc
            if link_domain and link_domain != domain:
                external_links_count += 1
        start_a_index = end_a_index + 1
    
    features['external_links_count'] = external_links_count

    # Count number of images
    features['images_count'] = content.count('<img')

    # Count number of script tags
    features['scripts_count'] = content.count('<script')

    # Length of visible text
    text_content = ' '.join(content.split('<')[0] for content in content.split('>')[1::2])
    features['text_length'] = len(text_content)

    # Count number of forms
    features['forms_count'] = content.count('<form')

    # Check for login forms
    features['login_forms_count'] = content.count('type="password"')

    # Check for inline scripts
    features['inline_scripts_count'] = content.count('<script>')

    # Check for iframes
    features['iframes_count'] = content.count('<iframe')

    # Check for suspicious words in text content
    suspicious_words = ['prize', 'winner', 'free', 'congratulations', 'limited time', 'offer']
    features['suspicious_words_count'] = sum(text_content.lower().count(word) for word in suspicious_words)

    # Check for email links
    features['email_links_count'] = content.count('mailto:')

    # Check for presence of analytics tools
    features['has_google_analytics'] = int('www.google-analytics.com' in content)

    # Check for redirects
    features['redirect_count'] = len(response.history)
    
    return features

'''
url = "http://degenalgo.art"
features = get_url_features(url)
print(features)

14 features
'''

data = [
("http://www.bbc.com", 0),
("http://www.cnn.com", 0),
("http://www.nytimes.com", 0),
("http://www.theguardian.com", 0),
("http://www.reuters.com", 0),
("http://www.npr.org", 0),
("http://www.forbes.com", 0),
("http://www.bloomberg.com", 0),
("http://www.wsj.com", 0),
("http://www.usatoday.com", 0),
("http://www.google.com", 0),
("http://www.bing.com", 0),
("http://www.yahoo.com", 0),
("http://www.duckduckgo.com", 0),
("http://www.baidu.com", 0),
("http://www.facebook.com", 0),
("http://www.twitter.com", 0),
("http://www.linkedin.com", 0),
("http://www.instagram.com", 0),
("http://www.pinterest.com", 0),
("http://www.wikipedia.org", 0),
("http://www.khanacademy.org", 0),
("http://www.coursera.org", 0),
("http://www.edx.org", 0),
("http://www.udacity.com", 0),
("http://www.techcrunch.com", 0),
("http://www.wired.com", 0),
("http://www.theverge.com", 0),
("http://www.arstechnica.com", 0),
("http://www.cnet.com", 0),
("http://www.amazon.com", 0),
("http://www.ebay.com", 0),
("http://www.walmart.com", 0),
("http://www.bestbuy.com", 0),
("http://www.target.com", 0),
("http://www.bankofamerica.com", 0),
("http://www.wellsfargo.com", 0),
("http://www.chase.com", 0),
("http://www.paypal.com", 0),
("http://www.mint.com", 0),
("http://www.usa.gov", 0),
("http://www.un.org", 0),
("http://www.who.int", 0),
("http://www.nasa.gov", 0),
("http://www.cdc.gov", 0),
("http://www.webmd.com", 0),
("http://www.mayoclinic.org", 0),
("http://www.healthline.com", 0),
("http://www.medlineplus.gov", 0),
("http://www.clevelandclinic.org", 0),
("http://www.netflix.com", 0),
("http://www.hulu.com", 0),
("http://www.spotify.com", 0),
("http://www.youtube.com", 0),
("http://www.imdb.com", 0),
("http://www.expedia.com", 0),
("http://www.tripadvisor.com", 0),
("http://www.airbnb.com", 0),
("http://www.booking.com", 0),
("http://www.kayak.com", 0),
("http://www.indeed.com", 0),
("http://www.monster.com", 0),
("http://www.glassdoor.com", 0),
("http://www.linkedin.com/jobs", 0),
("http://www.simplyhired.com", 0),
("http://www.reddit.com", 0),
("http://www.quora.com", 0),
("http://www.stackexchange.com", 0),
("http://www.medium.com", 0),
("http://www.github.com", 0), 
("http://www.injex.top", 1),
("http://www.naka-wallet.com", 1),
("http://www.tbmcoats.com", 1),
("http://www.pancakaswap.club", 1),
("http://www.solcelty.com", 1),
("http://www.debugnodes.com", 1),
("http://www.dopewallet.in", 1),
("http://www.dogto.io", 1),
("http://www.allo.ltd", 1),
("http://www.fuwum.com", 1),
("http://www.jkl.bitonite-copytrade.shop", 1),
("http://www.grokcoinmeme.net", 1),
("http://www.abnbclaim.com", 1),
("http://www.bafybeicztj6d2jb7rq3bkhk6qenpgujniiqeyzawsylqrw2nawybbhixpu.ipfs.dweb.link", 1),
("http://www.secure2024wallet7720user2987635mask24.laviewddns.com", 1),
("http://www.sandbox.com.lc", 1),
("http://www.c1aim-etherfi.com", 1),
("http://www.trumpnft.buzz", 1),
("http://www.wienerdog-io.com", 1),
("http://www.claim-neo.org", 1),
("http://www.offer-listing-support.app.smwgadget.com", 1),
("http://www.support-my-offer.app.mibgroup.us", 1),
("http://www.rewards-mendiflnance.app", 1),
("http://www.refund-magepieprotocol.com", 1),
("http://www.trailblazerstaiko.pages.dev", 1),
("http://www.foundationsrewards.com", 1),
("http://www.blum.icu", 1),
("http://www.voting-penpie.net", 1),
("http://www.hamsterkombt.xyz", 1),
("http://www.gasfeehashrate.vip", 1),
("http://www.rndrreward.org", 1),
("http://www.pulsex-app.io", 1),
("http://www.beta-goneuraiai.com", 1),
("http://www.sebycoin.xyz", 1),
("http://www.greenbtclive.pages.dev", 1),
("http://www.revokecashsafe.xyz", 1),
("http://www.join-overlord.xyz", 1),
("http://www.ailocate-eigenpoints.net", 1),
("http://www.monad-snapshot.net", 1),
("http://www.notcloud.biz", 1),
("http://www.claim.elher.io", 1),
("http://www.finishworkdapp.pages.dev", 1),
("http://www.claimcybro.com", 1),
("http://www.foundations-ether.fi", 1),
("http://www.registration-swellnetwork.org", 1),
("http://www.galxequest.app", 1),
("http://www.wormhole-foundation.app", 1),
("http://www.ajucx.com", 1),
("http://www.multiversenodes.pages.dev", 1),
("http://www.axiesinfinity.co", 1),
("http://www.baked-tokens.com", 1),
("http://www.dapp-wallet-connect.pages.dev", 1),
("http://www.munchieswithbrett.com", 1),
("http://www.connect-zklink.com", 1),
("http://www.app-galx.com", 1),
("http://www.lounge.finance", 1),
("http://www.rewards.pixelverses.app", 1),
("http://www.launch-zkfair.io", 1),
("http://www.tajko.eu", 1),
("http://www.babylone.evseen.com", 1),
("http://www.stake-devolvedai.com", 1),
("http://www.golddex-web3.cc", 1),
("http://www.rewards-ultiverse.app", 1),
("http://www.web3pad.pages.dev", 1),
("http://www.ulteriori-informazioni.com", 1),
("http://www.lxp-register.com", 1),
("http://www.anyonemigration.online", 1),
("http://www.claim.retik-dashboard.com", 1),
("http://www.wishtomi.shop", 1),
("http://www.synchrony.meta-coinpod.app", 1),
("http://www.claimpepe.exchange", 1),
("http://www.pepenewcoin.buzz", 1),
("http://www.signup-prevasea.events", 1),
("http://www.enter-destra.com", 1),
("http://www.dusksnetwork.com", 1),
("http://www.etherfi-foundation.com", 1),
("http://www.hzucx.com", 1),
("http://www.claims-dusk.network", 1),
("http://www.join-movementiabs.app", 1),
("http://www.season2-roost.wtf", 1),
("http://www.thedefiant.click", 1),
("http://www.join-chimpers.xyz", 1),
("http://www.hnwt39-5000.csb.app", 1),
("http://www.multirelive.pages.dev", 1),
("http://www.bitgetzone.cc", 1),
("http://www.quadntizedex3c.me", 1),
("http://www.twmax.win", 1),
("http://www.coodeia.com", 1),
("http://www.quantizebert.net", 1),
("http://www.m.budafay.cc", 1),
("http://www.upbitmarket.com", 1),
("http://www.nttdatadex.cc", 1),
("http://www.bkexgroup.top", 1),
("http://www.uniswapglobal.top", 1),
("http://www.uniswap-global.com", 1),
("http://www.coinupex-market.com", 1),
("http://www.geminizone.top", 1),
("http://www.bitgetno1.top", 1),
("http://www.uniswap.vip", 1),
("http://www.bitgetno1.cc", 1),
("http://www.coinupex-market.top", 1),
("http://www.bitget-web3.top", 1),
("http://www.coinupex-market.vip", 1),
("http://www.upbitzone.com", 1),
("http://www.bitget-bitget.top", 1),
("http://www.geminizone.cc", 1),
("http://www.emc-group.cc", 1),
("http://www.curvemarket.vip", 1),
("http://www.m.web36dpi.net", 1),
("http://www.wosannain.com", 1),
("http://www.bacbaczone.top", 1),
("http://www.max-has.vip", 1),
("http://www.curveone.cc", 1),
("http://www.xrtryule.com", 1),
("http://www.quntzedx3.com", 1),
("http://www.qutzendx3.com", 1),
("http://www.qntzedx.com", 1),
("http://www.arnadey.net", 1),
("http://www.quantizeide.com", 1),
("http://www.base-tax.xyz", 1),
("http://www.datsphre.com", 1),
("http://www.budainole.com", 1),
("http://www.m.hyrouvane.cc", 1),
("http://www.uniswapglobal.cc", 1),
("http://www.hitbtsg.com", 1),
("http://www.haubergeo.net", 1),
("http://www.eamelia.cc", 1),
("http://www.claim-kintoxyz.app", 1),
("http://www.engberta.com", 1),
("http://www.xrtrading.co", 1),
("http://www.shakingas.com", 1),
("http://www.hitbtcot.me", 1),
("http://www.qutantizedxs.com", 1),
("http://www.web36uv.com", 1),
("http://www.webbkit.org", 1),
("http://www.cryptojona.org", 1),
("http://www.gatepol.com", 1),
("http://www.bitgetcoin.win", 1),
("http://www.budaned.com", 1),
("http://www.m.taxiwayi.cc", 1),
("http://www.reeboks.ink", 1),
("http://www.ethfinance.vip", 1),
("http://www.bitbitzone1.live", 1),
("http://www.quantizedex3.com", 1),
("http://www.tonier.ink", 1),
("http://www.m.taxiwayi.com", 1),
("http://www.urtimor.com", 1),
("http://www.bitbitzone.site", 1),
("http://www.drbudain.com", 1)
]
train_urls = [item[0] for item in data]
train_labels = [item[1] for item in data]

# Assuming get_url_features function is defined as shown previously
features_list = []

for url in train_urls:
    features = get_url_features(url)
    features_list.append(features)

# Convert features to a numpy array
feature_names = list(features_list[0].keys())
print(feature_names)
features_array = np.array([[features[name] for name in feature_names] for features in features_list])

# Convert labels to a numpy array
labels_array = np.array(train_labels)


dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
batch_size = 2
dataset = dataset.shuffle(len(features_array)).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(14, activation='relu', input_shape=(len(feature_names),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(dataset, epochs=epochs)

# Evaluate the model (optional)
loss, accuracy = model.evaluate(dataset)
print(f"Loss: {loss}, Accuracy: {accuracy}")

#model.save("scamFinder.h5")
tfjs.save_keras_model(model,"./tjfs_files")