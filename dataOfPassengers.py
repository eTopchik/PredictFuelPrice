import csv
import requests
import os
import xlsxwriter
from bs4 import BeautifulSoup


def paste(url, attr, z, y):
    source_code = requests.get(url).text # going to the website
    soup = BeautifulSoup(source_code, 'lxml') # represents the document as a nested data structure
    rows = soup.findAll(z, attr) # finds all lines with tr and class : dataTDRight
    for row in rows:
        cols = row.find_all(y)
        cols = [x.text.strip() for x in cols] # converts data as text
        if cols[1] == str(7):  # takes data for July
            print(cols[2][:2]) # prints the number of millions
        with open('amount.csv', 'a', newline='') as file:  # appending data to file
            if cols[0] > str(2003) and cols[1] == str(7) : # July since 2004
                csv.writer(file).writerow(cols[2].split(',')[:1])  # as the number is millions i want to take only
                                                            # number which describes how many them and write to the file
    price = open('price.csv', 'r') # reads data from file
    amount = open('amount.csv', 'r')
    workbook = xlsxwriter.Workbook('UsaFuelPassenger.xlsx') # stores file
    worksheet = workbook.add_worksheet("My sheet") # stores worksheet
    row = 0
    col = 0
    for name1 in price:
        worksheet.write(row, col, name1) # writes data to 1st column (A)
        row += 1 # moves to the next row
    row = 0
    for name2 in amount:
        worksheet.write(row, col + 1, name2) # writes data to 2nd column (B)
        row += 1
    workbook.close() # closed and stopped work with file

paste('https://www.transtats.bts.gov/Data_Elements.aspx?Data=1', {'class': "dataTDRight"}, "tr", "td")
