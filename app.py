from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from ta.trend import MACD
from ta.momentum import RSIIndicator
import tensorflow as tf
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.trend import EMAIndicator
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

app = Flask(__name__)

# Load model dan scaler untuk saham US
rf_model_us = joblib.load('models/random_forest_model.joblib')
lstm_model_us = load_model('models/lstm_model.h5')
gru_model_us = load_model('models/gru_model.keras')
scaler_us = joblib.load('models/scaler.joblib')
price_scaler_us = joblib.load('models/price_scaler.joblib')
feature_scaler_us = joblib.load('models/feature_scaler.joblib')

# Load model dan scaler untuk saham Indonesia
rf_model_id = joblib.load('models/random_forest_model_sahamlocal.joblib')
lstm_model_id = load_model('models/lstm_model_sahamlocal.h5')
gru_model_id = load_model('models/gru_model_sahamlocal.keras')
scaler_id = joblib.load('models/scaler_sahamlocal.joblib')
price_scaler_id = joblib.load('models/price_scaler_sahamlocal.joblib')
feature_scaler_id = joblib.load('models/feature_scaler_sahamlocal.joblib')

def stock_data(symbol, period='2y'):
    try:
        # Tambah validasi simbol
        if not symbol:
            raise ValueError("Kode saham tidak boleh kosong")
            
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"Tidak ada data untuk saham {symbol}")
            
        # Hitung indikator teknikal
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Signal_Line'] = macd.macd_signal()
        
        # RSI
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # EMA
        ema20 = EMAIndicator(df['Close'], window=20)
        ema50 = EMAIndicator(df['Close'], window=50)
        df['EMA20'] = ema20.ema_indicator()
        df['EMA50'] = ema50.ema_indicator()
        
        #PVB
        df['PVB'] = df['Close'] * df['Volume']
        
        # Tambahan fitur
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        
        df = df.fillna(method='ffill')
        if df.isnull().any().any():  
            df = df.fillna(method='bfill')
            
        if df.isnull().any().any():  
            raise ValueError(f"Data tidak lengkap untuk saham {symbol}")
            
        return df
        
    except Exception as e:
        print(f"Error getting stock data for {symbol}: {str(e)}")
        return None
    
    
def predict_signal(data, rf_model, lstm_model, scaler):
    try:
        features = ['Close', 'RSI', 'MACD', 'MACD_Signal_Line']
        # Normalisasi data
        data_scaled = scaler.transform(data[features])
        
        # Prediksi dengan Random Forest
        rf_pred = rf_model.predict(data_scaled)
        
        # Prediksi dengan LSTM
        data_lstm = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))
        lstm_pred = lstm_model.predict(data_lstm)
        lstm_pred = np.argmax(lstm_pred, axis=1)
        
        # Gabungkan prediksi
        final_pred = []
        for rf, lstm in zip(rf_pred, lstm_pred):
            if rf == lstm:
                final_pred.append(rf)
            else:
                # Gunakan prediksi Random Forest jika berbeda
                final_pred.append(rf)
        
        # Konversi ke label
        label_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        return [label_map[p] for p in final_pred]
    except Exception as e:
        print(f"Error in predict_signal: {str(e)}")
        raise
def data_augmentation(X, y):
    X_aug, y_aug = X.copy(), y.copy()
    
    
    noise_factor = 0.002
    X_noise = X + np.random.normal(0, noise_factor, X.shape)
    y_noise = y + np.random.normal(0, noise_factor / 2, y.shape)
    
    X_aug = np.concatenate([X_aug, X_noise])
    y_aug = np.concatenate([y_aug, y_noise])
    
    return X_aug, y_aug

def predict_price(data, gru_model, price_scaler, feature_scaler, lookback=60):
    try:
        price_features = [
            'Close', 'RSI', 'MACD', 'MACD_Signal_Line',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Stoch_K', 'Stoch_D', 'EMA20', 'EMA50'
        ]
        
        if len(data) < lookback:
            raise ValueError(f"Data tidak cukup, minimal {lookback} data point")
            
       
        df = data.copy()
        
        
        last_sequence = df[price_features].tail(lookback)
        
        # Normalisasi data
        sequence_scaled = feature_scaler.transform(last_sequence)
        sequence_reshaped = sequence_scaled.reshape((1, lookback, len(price_features)))
        
        
        X_aug, _ = data_augmentation(sequence_reshaped, np.zeros((1, 1)))  # Dummy target karena kita hanya butuh X_aug
        
        # Prediksi dengan GRU (gunakan augmented data)
        predictions = gru_model.predict(X_aug, verbose=0) 
        
        
        avg_prediction = np.mean(predictions, axis=0)
        
        # Inverse transform untuk mendapatkan harga asli
        predicted_price = price_scaler.inverse_transform(avg_prediction.reshape(-1, 1))[0][0]
        
        return predicted_price
        
    except Exception as e:
        print(f"Error in predict_price: {str(e)}")
        return data['Close'].iloc[-1]  # Jika gagal, gunakan harga terakhir


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker']
        market = request.form.get('market', 'us') 
        timeframe = request.form.get('timeframe', '1mo')
        
        # Tambahkan .JK untuk saham Indonesia jika belum ada
        if market == 'id' and not ticker.endswith('.JK'):
            ticker = f"{ticker}.JK"
            
       
        df_full = stock_data(ticker, period='2y')
        if df_full is None or df_full.empty:
            return jsonify({
                'error': f'Tidak dapat mengambil data untuk saham {ticker}. Pastikan kode saham benar.'
            })
            
        # Pastikan data cukup untuk analisis
        if len(df_full) < 60:  
            return jsonify({
                'error': f'Data tidak cukup untuk saham {ticker}. Diperlukan minimal 60 hari data.'
            })
            
        # Filter data sesuai timeframe
        df = df_full.copy()
        if timeframe == '1mo':
            df = df.last('30D')
        elif timeframe == '3mo':
            df = df.last('90D')
        elif timeframe == '6mo':
            df = df.last('180D')
        elif timeframe == '1y':
            df = df.last('365D')
        
        # Pilih model berdasarkan market
        try:
            if market == 'us':
                rf_model = rf_model_us
                lstm_model = lstm_model_us
                gru_model = gru_model_us
                scaler = scaler_us
                price_scaler = price_scaler_us
                feature_scaler = feature_scaler_us
            else:
                rf_model = rf_model_id
                lstm_model = lstm_model_id
                gru_model = gru_model_id
                scaler = scaler_id
                price_scaler = price_scaler_id
                feature_scaler = feature_scaler_id
                
            # Prediksi signal trading
            try:
                trading_signals = predict_signal(df.tail(5), rf_model, lstm_model, scaler)
            except Exception as e:
                print(f"Error in trading signal prediction: {str(e)}")
                trading_signals = ['Hold'] * 5  # Default ke Hold jika prediksi gagal
            
            # Prediksi harga
            try:
                price_prediction = predict_price(df_full, gru_model, price_scaler, feature_scaler)
            except Exception as e:
                print(f"Error in price prediction: {str(e)}")
                price_prediction = df['Close'].iloc[-1]  # Gunakan harga terakhir jika prediksi gagal
                
            # Siapkan response
            chart_data = {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'close': df['Close'].tolist(),
                'volume': df['Volume'].tolist(),
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'technical_indicators': {
                    'ema20': df['EMA20'].tolist(),
                    'ema50': df['EMA50'].tolist(),
                    'rsi': df['RSI'].tolist(),
                    'macd': df['MACD'].tolist(),
                    'macd_signal': df['MACD_Signal'].tolist(),
                    'bb_upper': df['BB_upper'].tolist(),
                    'bb_lower': df['BB_lower'].tolist(),
                    'pvb': df['PVB'].tolist()

                },
                'last_5_predictions': {
                    'dates': df.index[-5:].strftime('%Y-%m-%d').tolist(),
                    'prices': df['Close'].tail(5).round(2).tolist(),
                    'signals': trading_signals[-5:]
                },
                'price_prediction': round(float(price_prediction), 2),
                'currency': 'Rp ' if market == 'id' else '$',
                'market': market
            }
            
            return jsonify({
                'success': True,
                'data': chart_data,
                'message': f'Prediksi berhasil untuk {ticker}'
            })
            
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            return jsonify({
                'error': f'Terjadi kesalahan saat memproses prediksi: {str(e)}'
            })
            
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': f'Terjadi kesalahan: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True) 