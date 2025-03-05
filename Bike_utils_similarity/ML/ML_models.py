# from utils.scoring import get_res
# from utils.time_utils import get_TimeStamp_str
# from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import catboost
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from ..ML.exp_conf import cat_features

model_lst = [
             RandomForestRegressor(),
             BaggingRegressor(),
             catboost.CatBoostRegressor(iterations=1000, verbose=0, cat_features=cat_features),
             # LinearRegression(),
             # Lasso(),
             # Ridge(),
             # ElasticNet(),
             # BaggingRegressor(),
             # AdaBoostRegressor()
            ]



