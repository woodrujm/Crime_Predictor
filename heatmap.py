from eda import make_df, serious_offenses_df, year_column

path = '/Users/hercules/desktop/galvanize/capstone/data/sf_crime.csv'
df = make_df(path)
df = year_column(df)

df = df[['resolution','category','location','year']]

serious_offenses = ["LARCENY/THEFT", "ASSAULT", "VEHICLE THEFT", "DRUG/NARCOTIC", "VANDALISM", "BURGLARY", "MISSING PERSON", "ROBBERY", "WEAPON LAWS", "STOLEN PROPERTY", "SEX OFFENSES", "FORCIBLE", "KIDNAPPING", "ARSON"]

df_serious_offenses = serious_offenses_df(df, serious_offenses)
#

# # create dataframe for violent_offenses
# would be better to do it as a dictionary
date_min = df.year.min()
date_max = df.year.max()
longlats_dict = {}
for date in xrange(date_min,date_max+1):
    # longlats[date] = df[df.year == date] # make this work
    exec("df_seriousoffenses_{} = df_serious_offenses[df_serious_offenses['year']==date]".format(date))

## make a dictionary instead of separate dataframe and loop over the dictionary keys as the year
# for year in xrange(2003,2018):
#     with open("longlats{}.txt".format(year),"w") as f:
#         for longlat in df_seriousoffenses_{}.location.format(year):
#             f.write("new google.maps.LatLng{}\n".format(longlat))


#changed df to name from local
with open("resolved2017.txt","w") as f:
    for longlat in l2017.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2016.txt","w") as f:
    for longlat in df_seriousoffenses_2016.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2015.txt","w") as f:
    for longlat in df_seriousoffenses_2015.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2015.txt","w") as f:
    for longlat in df_seriousoffenses_2015.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2014.txt","w") as f:
    for longlat in df_seriousoffenses_2014.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))


with open("longlats2013.txt","w") as f:
    for longlat in df_seriousoffenses_2013.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2012.txt","w") as f:
    for longlat in df_seriousoffenses_2012.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2011.txt","w") as f:
    for longlat in df_seriousoffenses_2011.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2010.txt","w") as f:
    for longlat in df_seriousoffenses_2010.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))

with open("longlats2009.txt","w") as f:
    for longlat in df_seriousoffenses_2009.location:
        f.write( "new google.maps.LatLng{},\n".format(longlat))
