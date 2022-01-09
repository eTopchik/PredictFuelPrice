from bs4 import BeautifulSoup
import requests
import csv


def take_data(url, year, month):
    source_code = requests.get(url).text # going to the website
    soup = BeautifulSoup(source_code, "lxml") # represents the document as a nested data structure
    rows = soup.find_all('tr')[1:]  # finds all lines with tr , as the first is empty , we take from 2nd row
    for row in rows:  # iterating over each row
        cols = row.find_all('td', {'class': "dataTD"}) + row.find_all('td', {'class': "dataTDRight"})
        # takes data which is necessary
        cols = [x.text.strip() for x in cols] # converts data as text
        if month in cols:  # takes data for given month
            if int(cols[0]) >= year:  # data from given year up to last one
                print(float(cols[4]))
        with open('price.csv', 'a', newline='') as file:  # appending data to file
            if month in cols:
                if int(cols[0]) >= year:
                    csv.writer(file).writerow(cols[4:5])  # writes data to file from 4th column


take_data('https://www.transtats.bts.gov/fuel.asp?pn=1', 2004, "July") # calling the function with given parameters
