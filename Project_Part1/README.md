
The structure of this Part 1 of Project is:

- airbnb folder: contains Scrapy framework for Airbnb listing scraping
	- scrapy.cfg: config file automatically created by Scrapy startproject
	- airbnb folder: project's Python module
		- spiders folder: contains spiders
			- airbnb_spider.py: project spider definitions
		- items.py: project items definition file
		- middlewares: project middlewares
		- pipelines.py: project pipelines
		- settings.py: project settings

- hotels folder: contains scraping script for Hotels.com and data
	- tripadv_scrape.py: scraping script for Hotels.com
	- tripadv_prices.json: first version raw output from scraping
	- tripadv_prices.csv: final version raw output from scraping
	- chromedriver: required for scraping

- utilities folder: contains API request script for Walkscore API and Google Maps API
	- walkscore.py: API request script for Walkscore
	- GoogleMapRetrieve.py: API request script for Google Maps

- yelp folder: contains scraping script for Yelp.com
	- yelp.py: scraping script for Yelp.com
	- clean.py: script for putting together results and reformat
	- YelpData.csv: output from scraping
	- clean_data.csv: reformatted output from scraping
	- Zipcode.txt: list of zip codes in NYC
	
- Airbnb_listings.csv: scraped output after metadata clean
- airbnb_metadata_clean.py: script for metadata clean
- DataCleaning-Part1.py: script for determine the cleanness of all raw data
- Missing&Incorrect_Airbnb.txt: output of DataCleaning-Part1.py
- Missing&Incorrect_HotelPrice.txt: output of DataCleaning-Part1.py
- Missing&Incorrect_YelpData.txt: output of DataCleaning-Part1.py
- Project 1.docx: Project 1 Part 1 documentation
