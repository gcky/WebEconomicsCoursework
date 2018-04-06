
import pandas as pd


def main():
    train_file = pd.read_csv('train.csv')
    train_file["advertiser"].value_counts()
    train_file["click"].value_counts()
    data.groupby('Agent', {'CTR': data.aggregate.average('Click')})



def advertiser():
    # Total clicks
    advertiser_score = advertiser_score.join(pd.DataFrame(
    {'clicks': train_file[train_file['click'] == 1].groupby('advertiser').size()}).reset_index(drop=True))
    # Impressions
    advertiser_score = pd.DataFrame({'impressions': train_file.groupby('advertiser').size()}).reset_index()
    # Click-Through
    advertiser_score['CTR'] = advertiser_score['clicks']/advertiser_score['impressions'] * 100
    # CPC
    advertiser_score['CPC'] = advertiser_score['cost']/advertiser_score['clicks']
    # CPM
    advertiser_score['CPM'] = advertiser_score['cost']*1000/advertiser_score['impressions']

    print(advertiser_score)


def dailyscore():
    # Total Clicks
    day_score = day_score.join(pd.DataFrame(
    {'Clicks': train_file[train_file['click'] == 1].groupby('weekday').size()}).reset_index(drop=True))
    # Impressions
    day_score = pd.DataFrame({'impressions': train_file.groupby('weekday').size()}).reset_index()
    # Click-Through
    day_score['CTR'] = day_score['clicks'] / day_score['impressions'] * 100
    # CPC
    day_score['CPC'] = day_score['cost'] / day_score['Clicks']
    # CPM
    day_score['CPM'] = day_score['cost'] * 1000 / day_score['impressions']
    # Cost per day
    day_score = day_score.join(pd.DataFrame({'cost': train_file.groupby(
                                                                                                           ['weekday'])['payprice'].sum()}).reset_index(drop=True)/1000)
    print(day_score)

if __name__ == "__main__":
    
    main()
    advertiser()
    dailyscore()
    click_stats = train_file[train_file['Click'] == 1]
    click_stats.groupby('weekday').size()
    click_stats.groupby('advertiser').size()
    print(click_stats)
