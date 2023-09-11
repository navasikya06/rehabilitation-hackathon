import streamlit as st
import pandas as pd
import time


import pickle as pk

# classification models
week2 = pk.load(open('./models/lor1.pkl','rb'))
week6 = pk.load(open('./models/lor2.pkl','rb'))
week12 = pk.load(open('./models/lor3.pkl','rb'))
week26 = pk.load(open('./models/lor4.pkl','rb'))
week52 = pk.load(open('./models/lor5.pkl','rb'))

# regression models
net_total_model = pk.load(open('./models/xgboost-nettotal.pkl','rb'))
total_cost_model = pk.load(open('./models/xgboost-totalcost.pkl','rb'))
other_paid_model = pk.load(open('./models/xgboost-otherpaid.pkl','rb'))
payment_medical_model = pk.load(open('./models/xgboost-paymentmedical.pkl','rb'))
other_paid_risk_model = pk.load(open('./models/xgboost-otherpaidrisk.pkl','rb'))
payment_legaldef_model = pk.load(open('./models/xgboost-paymentlegaldef.pkl','rb'))
payment_legalplaintiff_model = pk.load(open('./models/xgboost-paymentlegalpaintiff.pkl','rb'))
payment_impairment_model = pk.load(open('./models/xgboos-paymentimpairment.pkl','rb'))
payment_investigation_model = pk.load(open('./models/xgboost-paymentinvestigation.pkl','rb'))
   

def predict(df):
    prediction_df = pd.DataFrame(
        {
            'Fitness_week2': week2.predict(df)[0],
            'Fitness_week6': week6.predict(df)[0],
            'Fitness_week12': week12.predict(df)[0],
            'Fitness_week26': week26.predict(df)[0],
            'Fitness_week52': week52.predict(df)[0],
            'Net_total_incurred': net_total_model.predict(df)[0],
            'Total_Paid': total_cost_model.predict(df)[0],
            'Other_Paid': other_paid_model.predict(df)[0],
            'Payment_medical': payment_medical_model.predict(df)[0],
            'Other_paid_risk': other_paid_risk_model.predict(df)[0],
            'Payment_legal_defendant': payment_legaldef_model.predict(df)[0],
            'Payment_legal_plaintiff': payment_legalplaintiff_model.predict(df)[0],
            'Payment_Impairment': payment_impairment_model.predict(df)[0],
            'Payment_investigation_surveillance': payment_investigation_model.predict(df)[0]
        }
    )
    new_df = pd.concat([df, prediction_df], axis=1)
    return new_df

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')



def main():
    st.title("Cost Prediction Model \nMacquarie University Hackathon for Navigator Group")

    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type='xlsx')

    if uploaded_file is not None:
        org_df = pd.read_excel(uploaded_file)
        new_df = predict(org_df)

        my_bar = st.progress(0, text='Processing')

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text="Processing")

        st.write(new_df)

        st.download_button(
            label="Download result file",
            data=convert_df(new_df),
            file_name='result.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()