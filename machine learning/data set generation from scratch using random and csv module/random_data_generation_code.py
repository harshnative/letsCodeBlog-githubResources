#!/usr/bin/env python
# coding: utf-8

# In[355]:


import numpy
import pandas


# In[356]:


myData = pandas.DataFrame() 

myData


# In[357]:


# count of the current col
count = 0


# In[358]:


# inserting the roll no col
# index to insert , col name , what to insert
myData.insert(count , "rollNo" , list(range(100000 , 100000 + 50000)))

# icreamenting the count so that we will insert next col in next index
count = count + 1

myData


# In[359]:


# importing names
namesDf = pandas.read_csv("names.csv")
namesDf


# In[360]:



# inserting the names 
myData.insert(count , "first name" , namesDf["first name"])

count = count + 1


# In[361]:


myData


# In[362]:


# making a list of first names
import random

firstNameList = list(namesDf["first name"])


# In[363]:


# names were only 37000 we needed 50000
# so were are insert random names from he names list above in the cols having nan value
for i in myData.index:
    
    # to check if the dataframe col is Nan
    if(pandas.isnull(myData.at[i, "first name"])):
        myData.at[i, "first name"] = random.choice(firstNameList)


# In[364]:


myData


# In[365]:



# inserting the last names 
myData.insert(count , "last name" , namesDf["last name"])

count = count + 1


# In[366]:


myData


# In[367]:


# making a list of first names

lastNameList = list(namesDf["last name"])


# In[368]:


# names were only 37000 we needed 50000
# so were are insert random names from he names list above in the cols having nan value
for i in myData.index:
    
    # to check if the dataframe col is Nan
    if(pandas.isnull(myData.at[i, "last name"])):
        myData.at[i, "last name"] = random.choice(lastNameList)


# In[369]:


myData


# In[370]:


# inserting full name col
myData.insert(count , "fullName" , [i + j for i,j in zip(myData["first name"] , myData["last name"])])

# icreamenting the count so that we will insert next col in next index
count = count + 1

myData


# In[371]:


# adding branches col name
branches = ["CSE" , "COE" , "ECE" , "MEC" , "BOT", "ENC" , "CSE-MBA" , "ECE-MBA" , "MEC-MBA"]

randomList50000 = []
for i in range(50000):
    randomList50000.append(random.choice(branches))

myData.insert(count , "branch" , randomList50000)

count = count + 1

myData


# In[372]:


# inserting major list
CSE_COE_Majors = ["AI" , "Data Science" , "cloud" , "cyber"]
ECE_ENC_Majors = ["Micro electronic" , "IOT" , "Robotics"]

CSE_COE_Majors_applicables = ["CSE" , "COE" , "CSE-MBA"]
ECE_ENC_Majors_applicables = ["ECE" , "ENC" , "ECE-MBA"]

for i in myData.index:
    if(myData.at[i, "branch"] in CSE_COE_Majors_applicables):
        myData.at[i, "major"] = random.choice(CSE_COE_Majors)
    elif(myData.at[i, "branch"] in ECE_ENC_Majors_applicables):
        myData.at[i, "major"] = random.choice(ECE_ENC_Majors)
    else:
        myData.at[i, "major"] = numpy.nan
        
count = count + 1

myData


# In[373]:


# adding year col name
for i in myData.index:
        myData.at[i, "year"] = int(random.randint(1 , 4))
        
count = count + 1

myData


# In[374]:


# adding year col name
for i in myData.index:
        # semester can be either current year * 2 or current year * 2 - 1 
        myData.at[i, "sem"] = random.choice([(myData.at[i, "year"] * 2) - 1 , myData.at[i, "year"] * 2])
        
count = count + 1

myData


# In[375]:


# inserting fees col
fees = {
    "CSE" : 200000 , "COE" : 190000 , "ECE" : 170000 , "MEC" : 130000 , "BOT" : 140000, "ENC" : 185000 , "CSE-MBA" : 215000 , "ECE-MBA" : 200000 , "MEC-MBA" : 180000
}

for i in myData.index:
    myData.at[i, "fees"] = fees.get(myData.at[i, "branch"] , numpy.nan)
    
count = count + 1

myData


# In[376]:


# inserting total program fee col and program duration col

# MBA is of 5 years
for i in myData.index:
    if(str(myData.at[i, "branch"]).find("MBA") != -1):
       myData.at[i, "program duration"] = 5
       myData.at[i, "total program fee"] = myData.at[i, "fees"] * 5 * 2
    else:
       myData.at[i, "program duration"] = 4
       myData.at[i, "total program fee"] = myData.at[i, "fees"] * 4 * 2
    
count = count + 1

myData


# In[377]:


# inserting paid fee col

for i in myData.index:
       myData.at[i, "paid fee"] = bool(random.randint(0,1))
    
count = count + 1

myData


# In[378]:


# inserting CGPA col

for i in myData.index:
       myData.at[i, "CGPA"] = random.random() + random.randint(5 , 9)
    
count = count + 1

myData


# In[379]:


# inserting grade col

for i in myData.index:
    if(myData.at[i, "CGPA"] > 9):
        myData.at[i, "grade"] = "A"
    elif(myData.at[i, "CGPA"] > 8):
        myData.at[i, "grade"] = "A-"
    elif(myData.at[i, "CGPA"] > 7):
        myData.at[i, "grade"] = "B"
    elif(myData.at[i, "CGPA"] > 6):
        myData.at[i, "grade"] = "B-"
    else:
        myData.at[i, "grade"] = "C"
    
count = count + 1

myData


# In[381]:


# inserting hostel
import string

for i in myData.index:
        myData.at[i, "grade"] = random.choice(list(string.ascii_lowercase)[:15])
    
count = count + 1

myData


# In[382]:


# exporting to csv

myData.to_csv("studentsData.csv")


# In[ ]:




