### Crisis Help-Call Extraction — Datasets & Filtering

This repo lists the datasets we use and the filtering logic we apply to extract likely help-needed tweets after disasters. (Project aims/models will be added later.)

# Datasets

- HumAID (QCRI/CrisisNLP) — Human-labeled tweets across many disaster types. https://crisisnlp.qcri.org/humaid_dataset
- Turkey & Syria Earthquake Tweets (Kaggle, by swaptr) — Tweets related to the Feb 6, 2023 earthquakes. https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets
- CrisisBench (GitHub, firojalam) — Consolidated crisis datasets for benchmarking. https://github.com/firojalam/crisis_datasets_benchmarks

# Filtering logic (high level)

We flag tweets as potential help calls using text patterns in Turkish and English. The logic combines these signal groups:

SOS phrases (e.g., “yardım edin”, “acil yardım”, “need help”, “urgent help”, “trapped”, “send ambulance”).

Needs / medical / evacuation (e.g., “su lazım”, “çadır”, “evacuate”, “need water”, “insulin”, “oxygen”).

Locators: phone numbers, geographic coordinates, address tokens (e.g., mah./sok./no/apt.), or location words (“location”, “address”, “coordinates”).

Hazard cues: disaster terms such as earthquake, flood, wildfire, hurricane, landslide, avalanche, eruption, explosion and their Turkish equivalents.

Noise controls: exclude donation/announcement language (e.g., “donation”, “fundraiser”), drop link-only posts, and keep retweets only if they match the help patterns.