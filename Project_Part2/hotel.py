import pandas as pd

def clean_hotel(data):
    output1 = data.groupby('postalCode').count().reset_index()
    output1.columns = ['zipcode', 'count']
    output2 = data.groupby(['postalCode'])['price'].mean().reset_index()
    output2.columns = ['zipcode', 'averagePrice']
    output3 = data.groupby(['postalCode'])['price'].max().reset_index()
    output3.columns = ['zipcode', 'maxPrice']
    output4 = data.groupby(['postalCode'])['price'].min().reset_index()
    output4.columns = ['zipcode', 'minPrice']
    output = output1
    for df in [output2, output3, output4]:
        output = pd.merge(output, df, on='zipcode')
    return output

def main():
    data = pd.read_csv("./data/Hotel_prices.csv", usecols = ['postalCode', 'price'])
    data = data[data['postalCode'].apply(lambda x: len(str(x)) == 5)]
    #data['postalCode'] = data['postalCode'].astype(str)
    #data.info()
    clean_hotel(data).to_csv("hotel_data.csv")

main()