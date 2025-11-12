import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

with open('best_model.pkl', 'rb') as f:
    best_model=pkl.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf=pkl.load(f)
df=pd.read_csv('policy.csv')
st.title('Sentiment Analysis with Comparison')
user_input=st.text_area('Enter a review:')
if st.button('PREDICT'):
    if user_input.strip()=='':
        st.warning('Please enter some text for prediction.')
    else:
        input=tfidf.transform([user_input])
        prediction=best_model.predict(input)[0]
        sentiment_map={0:'negative', 1:'neutral', 2:'positive'}
        pre_sent=sentiment_map.get(prediction, str(prediction))
        st.write(f'**Predicted Setiment** {pre_sent}')
st.markdown("---")
st.header("Sentiment Analysis Insights")

with st.expander('1. what is the overall sentiment of user reviews?'):
    counts=df['sentiment'].value_counts(normalize=True).round(2)*100
    st.bar_chart(counts)

with st.expander('2. How does sentient vary by rating?'):
    if 'rating' in df.columns:
        chart_data = df['rating'].value_counts().sort_index()
        st.bar_chart(chart_data)
    else:
        st.info("Rating column not found")

with st.expander('3. Which keywords are most  associated with each sentiment class?'):
    for i  in ['negative', 'positive', 'neutral']:
        st.write(f'**{i.capitalize()} reviews word cloud**')
        text=' '.join(df[df['sentiment']==i]['review'].dropna().astype(str))
        if text.strip() == "":
            st.warning(f"No reviews available for **{i}** sentiment.")
            continue
        wc=WordCloud(width=600, height=400, background_color='white').generate(text)
        st.image(wc.to_array())
    
with st.expander('4. How has sentiment changed over time? '):
    if 'date' in df.columns:
        df['date']=pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df_time=df.dropna(subset=['date'])
        df_date=df_time.groupby(df_time['date'].dt.to_period('M'))['sentiment'].value_counts().unstack().dropna(axis=0)
        df_date.index=df_date.index.to_timestamp()
        st.line_chart(df_date)
    else:
        st.info('Data columns not listed')
    
with st.expander('5.Do verified users tend to leave more positive or negative reviews?'):
    if 'verified_purchase' in df.columns:
        df_customer=df.groupby('verified_purchase')['sentiment'].value_counts(normalize=True).unstack().dropna()
        st.bar_chart(df_customer)
    else:
        st.info('verified purchase column not found.')

with st.expander('6. Are longer reviews more likely to be negative or positive?'):
    fig, ax=plt.subplots()
    sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)
    st.pyplot(fig)

with st.expander('7. Which locations show the most positive or negative sentiment?'):
    if 'location' in df.columns:
        df_sen=df.groupby('location')['sentiment'].value_counts().unstack().fillna(0)
        st.bar_chart(df_sen)
    else:
        st.info('location column not found in')

with st.expander('8. Is there a difference in sentiment across platforms?'):
    if 'platform' in df.columns:
        df_plat=df.groupby('platform')['sentiment'].value_counts().unstack().fillna(0)
        st.bar_chart(df_plat)
    else:
        st.info('platform column not found')

with st.expander('9.Which ChatGPT versions are associated with higher/lower sentiment?'):
    if 'version' in df.columns:
        df_ver=df.groupby('version')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
        st.bar_chart(df_ver)
    else:
        st.info('version columns not found')

with st.expander('10. What are the most common negative feedback themes'):
    neg_rev=df[df['sentiment']=='negative']['review'].astype(str)
    neg_rev = neg_rev[neg_rev.str.strip() != ""]
    if neg_rev.empty:
        st.warning("⚠ No negative reviews found or all are empty!")
    else:
        vector=CountVectorizer(stop_words='english', max_features=20)
        x=vector.fit_transform(neg_rev)
        datafra=pd.DataFrame(x.toarray(), columns=vector.get_feature_names_out()).sum().sort_values(ascending=False)
        st.bar_chart(datafra)