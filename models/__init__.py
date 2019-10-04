import models.bi_double_lstm_gru
import models.bi_lstm_cnn
import models.bi_lstm_gru_attw_dense
import models.bi_lstm_gru_balanced
import models.bi_lstm_gru_selfatt_kfold
import models.bi_lstm_gru_spat_clr
import models.bi_lstm_gru_spat_clr_kfold
import models.text_cnn_att

MODEL_DICT = {
 'bi_double_lstm_gru': models.bi_double_lstm_gru,
 'bi_lstm_cnn': models.bi_lstm_cnn,
 'bi_lstm_gru_attw_dense': models.bi_lstm_gru_attw_dense,
 'bi_lstm_gru_balanced': models.bi_lstm_gru_balanced,
 'bi_lstm_gru_selfatt_kfold': models.bi_lstm_gru_selfatt_kfold,
 'bi_lstm_gru_spat_clr': models.bi_lstm_gru_spat_clr,
 'bi_lstm_gru_spat_clr_kfold': models.bi_lstm_gru_spat_clr_kfold,
 'text_cnn_att': models.text_cnn_att
}