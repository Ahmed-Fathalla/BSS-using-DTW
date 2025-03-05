import geopy.distance
def get_distance(key):
    '''
    Resources:
        # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    '''
    try:
        coords_1 = (key['Start Station Latitude'], key['Start Station Longitude'])
        coords_2 = (key['End Station Latitude'], key['End Station Longitude'])

        return geopy.distance.geodesic(coords_1, coords_2).m
    except:
        return -10

