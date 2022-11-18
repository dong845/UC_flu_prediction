import sys
import time
from scipy.ndimage import gaussian_filter1d
from utils import load_flu_cities_subset, load_flu_states, load_trends_states, load_trends_cities

sys.path.insert(0, 'models')
from persistence import persistance
from ar import ar
from ar_with_trends import ar_with_trends
from armulti import ar_multi
from armulti_with_trends import ar_multi_with_trends
from lstm import lstm
from lstm_with_trends import lstm_with_trends
from gru_gt_v1 import gru_with_trends1
from gru_gt_v2 import gru_with_trends2
from gru_gt_v3 import gru_with_trends3
from gru_gt_v21 import gru_with_trends21
from gru_gt_v31 import gru_with_trends31
from attention_with_trends import att_with_trends
from attention_origin_trends import attn_with_trends_v0
import matplotlib.pyplot as plt

model_lookup = {
    # 'persistence':persistance,
    'forecasting_ar': ar,
    'forecasting_armulti': ar_multi,
    'forecasting_rf': ar_multi,
    'forecasting_lstm': lstm,
    'nowcasting_ar': ar_with_trends,
    'nowcasting_armulti': ar_multi_with_trends,
    'nowcasting_rf': ar_multi_with_trends,
    'nowcasting_lstm': lstm_with_trends,
    "nowcasting_gru_v1": gru_with_trends1,
    "nowcasting_gru_v2": gru_with_trends2,
    "nowcasting_gru_v21": gru_with_trends21,
    "nowcasting_gru_v3": gru_with_trends3,
    "nowcasting_gru_v31": gru_with_trends31,
    "nowcasting_attn": att_with_trends,
    "nowcasting_attn_origin": attn_with_trends_v0
}

start = time.time()

geogran = 'city'  # Can be 'state' or 'city'
th = 1  # Can be 1, 2, 4, or 8
# Can be any of the keys in the model_lookup dictionary above
model_name = 'nowcasting_lstm'
online_learning = False  # Can be True or False
#output_fname = 'new_results_state/' + model_name + '_' + str(th) + '_____'
output_fname = 'bla.json'
if geogran == 'state':
    df = load_flu_states()
    df_trends = load_trends_states()
else:
    df = load_flu_cities_subset()   # (338, 159)
    df_trends = load_trends_cities()
model = model_lookup[model_name]
n_test = int(sys.argv[1])

def plot_lines():
    model7 = model_lookup["nowcasting_gru_v21"]
    _, _, history7 = model7(df, df_trends, th, n_test, True)
    model8 = model_lookup["nowcasting_gru_v31"]
    _, _, history8 = model8(df, df_trends, th, n_test, True)
    model3 = model_lookup["nowcasting_gru_v2"]
    _, _, history3 = model3(df, df_trends, th, n_test, True)
    model4 = model_lookup["nowcasting_gru_v3"]
    _, _, history4 = model4(df, df_trends, th, n_test, True)
    model5 = model_lookup["nowcasting_attn"]
    _, _, history5 = model5(df, df_trends, th, n_test, True)
    model6 = model_lookup["nowcasting_attn_origin"]
    _, _, history6 = model6(df, df_trends, th, n_test, True)
    model1 = model_lookup["nowcasting_lstm"]
    _, _, history1 = model1(df, df_trends, th, n_test, True)
    model2 = model_lookup["nowcasting_gru_v1"]
    _, _, history2 = model2(df, df_trends, th, n_test, True)
    
    epochs = list(range(1, 501))
    plt.plot(epochs,gaussian_filter1d(history1.history['loss'], sigma=3),'blue',label='GRU + GT')
    plt.plot(epochs,gaussian_filter1d(history2.history['loss'], sigma=3),'red',label='GRU + Optimized GT (v1)')
    plt.plot(epochs,gaussian_filter1d(history3.history['loss'], sigma=3),'yellow',label='GRU + Optimized GT (v2)')
    plt.plot(epochs,gaussian_filter1d(history4.history['loss'], sigma=3),'green',label='GRU + Optimized GT (v3)')
    plt.plot(epochs,gaussian_filter1d(history6.history['loss'], sigma=3),'purple',label='Optimized Model + GT')
    plt.plot(epochs,gaussian_filter1d(history5.history['loss'], sigma=3),'orange',label='Optimized Model + Optimized GT (v2)')
    plt.plot(epochs,gaussian_filter1d(history7.history['loss'], sigma=3),'brown',label='GRU + Optimized GT (v2+Separate)')
    plt.plot(epochs,gaussian_filter1d(history8.history['loss'], sigma=3),'pink',label='GRU + Optimized GT (v3+Separate)')
    plt.title('Training Loss (1)')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.plot(epochs,gaussian_filter1d(history1.history['val_loss'], sigma=3),'blue',label='GRU + GT')
    plt.plot(epochs,gaussian_filter1d(history2.history['val_loss'], sigma=3),'red',label='GRU + Optimized GT (v1)')
    plt.plot(epochs,gaussian_filter1d(history3.history['val_loss'], sigma=3),'yellow',label='GRU + Optimized GT (v2)')
    plt.plot(epochs,gaussian_filter1d(history4.history['val_loss'], sigma=3),'green',label='GRU + Optimized GT (v3)')
    plt.plot(epochs,gaussian_filter1d(history6.history['val_loss'], sigma=3),'purple',label='Optimized Model + GT')
    plt.plot(epochs,gaussian_filter1d(history5.history['val_loss'], sigma=3),'orange',label='Optimized Model + Optimized GT (v2)')
    plt.plot(epochs,gaussian_filter1d(history7.history['val_loss'], sigma=3),'brown',label='GRU + Optimized GT (v2+Separate)')
    plt.plot(epochs,gaussian_filter1d(history8.history['val_loss'], sigma=3),'pink',label='GRU + Optimized GT (v3+Separate)')
    plt.title('Test Loss (1)')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_lines()