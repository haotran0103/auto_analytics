import io
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

df = pd.read_excel('product.xlsx', engine='openpyxl')
df['total_revenue'] = df['Price'] * df['Historical sold']

def format_revenue(revenue):
    if revenue >= 1e9:
        return '{:,.2f} tỷ'.format(revenue / 1e9)
    elif revenue >= 1e6:
        return '{:,.2f} triệu'.format(revenue / 1e6)
    else:
        return '{:,.2f}'.format(revenue)

st.set_page_config(
    page_title='dash board',
    page_icon=':bar_chart:',
    layout='wide'
)

df1 = df[df['total_revenue'] != 0]
selected_tab = st.sidebar.radio('Chọn tab:', ['Báo cáo doanh thu', 'Báo cáo đánh giá người dùng',
                                              'Tiền xử lí dữ liệu', 'Data Mining', 'Mô hình dự đoán','Phân Tích Mô Tả','Phân tích thống kê'])
selected_product = None

col1, col2, col3 = st.columns(3)

if selected_tab == 'Báo cáo doanh thu':
    selected_order = st.sidebar.selectbox('Chọn thứ tự doanh thu:', ['Cao nhất', 'Thấp nhất'])

    selected_top = st.sidebar.number_input('Chọn top doanh thu:', min_value=1, max_value=len(df1), value=10)

    if selected_order == 'Cao nhất':
        sorted_df = df1.nlargest(selected_top, 'total_revenue')
    else:
        sorted_df = df1.nsmallest(selected_top, 'total_revenue')

    st.write('')

    total_quantity_sold = df['Historical sold'].sum()
    total_revenue = (df['Price'] * df['Historical sold']).sum()
    total_inventory = df['stock'].sum()

    with col1:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>Số lượng bán ra</h3>'
            f'<h2>{total_quantity_sold:,.0f}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>Tổng doanh thu</h3>'
            f'<h2>{format_revenue(total_revenue)}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>hàng còn tồn kho</h3>'
            f'<h2>{total_inventory:,.0f}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    c1, c2 = st.columns((6, 4))

    with c1:
        st.write('')

        top_10_products = sorted_df.copy()
        top_10_products['top'] = range(1, len(top_10_products) + 1)

        fig = px.bar(
            data_frame=top_10_products,
            x='total_revenue',
            y='top',
            labels={'total_revenue': 'Doanh thu', 'Name': 'Số sản phẩm'},
            color='total_revenue',
            color_continuous_scale='Viridis_r',
            orientation='h',
            hover_data={'top': False, 'Name': True},
            title=f'top sản phẩm {selected_top} có doanh thu {selected_order.lower()}'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write('')
        stock_by_product = df.groupby('Scategory_product')['stock'].sum().reset_index()
        fig = px.pie(
            stock_by_product,
            values='stock',
            names='Scategory_product',
            title='Tỉ lệ hàng tồn kho theo tên sản phẩm'
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig, use_container_width=True)

elif selected_tab == 'Báo cáo đánh giá người dùng':
    df2 = df1[df1['like_count'] != 0]

    selected_order = st.sidebar.selectbox('chọn số liệu đánh giá:', ['Được ưa thích nhất', 'Không ổn'])

    selected_top = st.sidebar.number_input('Chọn top dữ liệu đánh giá:', min_value=1, max_value=len(df2), value=10)

    selected_rating = st.sidebar.selectbox('chọn số liệu đánh giá:', ['Rating 5', 'Rating 1'])

    if selected_order == 'Được ưa thích nhất':
        sorted_df = df2.nlargest(selected_top, 'like_count')
    else:
        sorted_df = df2.nsmallest(selected_top, 'like_count')

    if selected_rating == 'Rating 5':
        df_rating_sorted = df2.groupby('Scategory_product')['Rating 5'].sum().reset_index()
    else:
        df_rating_sorted = df2.groupby('Scategory_product')['Rating 1'].sum().reset_index()

    st.write('')
    df1 = df[df['Rating'] != 0]
    total_rating = df['Total Rating'].sum()
    total_like = df['like_count'].sum()
    mean_rating = df1['Rating'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>Tổng số Rating</h3>'
            f'<h2>{total_rating:,.0f}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>Tổng lượt thích</h3>'
            f'<h2>{total_like:,.0f}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.write(
            f'<div style="text-align:center; box-shadow: 5px 5px 10px #888888; padding: 20px;">'
            f'<h3>TB đánh giá</h3>'
            f'<h2>{round(mean_rating, 2)}/5</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    c1, c2 = st.columns((6, 4))
    with c1:
        st.write('')

        top_10_products = sorted_df.copy()
        top_10_products['top'] = range(1, len(top_10_products) + 1)

        fig = px.bar(
            data_frame=top_10_products,
            x='like_count',
            y='top',
            labels={'like_count': 'Số lượt thích', 'Name': 'Số sản phẩm'},
            color='like_count',
            color_continuous_scale='Viridis_r',
            orientation='h',
            hover_data={'top': False, 'Name': True},
            title=f'top sản phẩm {selected_top} được {selected_order.lower()}'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write('')
        fig = px.pie(
            df_rating_sorted,
            values=selected_rating,
            names='Scategory_product',
            title='Tỉ lệ Rating của từng loại'
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig, use_container_width=True)
elif selected_tab == 'Tiền xử lí dữ liệu':
    st.subheader("Tiền xử lí dữ liệu")

    target_column = None
    attribute_columns = None

    if target_column is None:
        target_column = st.selectbox("Chọn cột target", df.columns)

    if attribute_columns is None:
        attribute_columns = st.multiselect("Chọn cột thuộc tính", df.columns.drop(target_column))

    if target_column is not None and attribute_columns:
        if st.button("Tiền xử lí"):
            X = df[attribute_columns].copy()
            y = df[target_column]

            numerical_features = X.select_dtypes(include=['float', 'int']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            columns_to_label_encode = [col for col in categorical_features if X[col].nunique() > 20]
            columns_to_onehot_encode = [col for col in categorical_features if col not in columns_to_label_encode]

            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore') if len(columns_to_onehot_encode) > 0 else None

            if y.dtype == 'object':
                target_transformer = LabelEncoder()
                y_encoded = target_transformer.fit_transform(y)
            else:
                target_transformer = None
                y_encoded = y

            transformers = []
            if numerical_transformer is not None:
                transformers.append(('num', numerical_transformer, numerical_features))
            if categorical_transformer is not None:
                transformers.append(('cat', categorical_transformer, columns_to_onehot_encode)) 

            preprocessor = ColumnTransformer(transformers=transformers)

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])

            X_processed = pipeline.fit_transform(X)  

            transformed_feature_names = numerical_features
            if categorical_transformer is not None:
                transformed_feature_names += categorical_transformer.get_feature_names_out(columns_to_onehot_encode)  # Use columns_to_onehot_encode here

            preprocessed_df = pd.DataFrame(X_processed, columns=transformed_feature_names)
            preprocessed_df[target_column] = y_encoded

            st.write('Dữ liệu đã được tiền xử lí:')
            st.write(preprocessed_df)
            st.session_state['preprocessed_df'] = preprocessed_df
        else:
            st.write('DataFrame hiện tại:')
            st.write(df[attribute_columns + [target_column]])

elif selected_tab == 'Data Mining':
    st.subheader("Data Mining")

    if 'preprocessed_df' not in st.session_state:
        st.write("Vui lòng thực hiện tiền xử lí dữ liệu trước khi thực hiện Data Mining.")
    else:
        preprocessed_df = st.session_state['preprocessed_df']
        st.write('DataFrame đã được tiền xử lí:')
        st.write(preprocessed_df)

        corr_matrix = preprocessed_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Ma trận tương quan')
        st.pyplot(plt)

        num_columns = preprocessed_df.select_dtypes(include=['float', 'int']).columns
        plt.figure(figsize=(12, 10))
        for i, column in enumerate(num_columns):
            plt.subplot(3, 3, i+1)
            sns.histplot(data=preprocessed_df, x=column, kde=True)
            plt.xlabel(column)
            plt.ylabel('Số lượng')
        plt.tight_layout()
        st.pyplot(plt)

        X = preprocessed_df.iloc[:, :-1] 
        y = preprocessed_df.iloc[:, -1]  

        clf = RandomForestClassifier()
        clf.fit(X, y)

        importance = clf.feature_importances_
        indices = np.argsort(importance)[::-1]
        feature_names = X.columns[indices]

        st.write('Các thuộc tính quan trọng:')
        for i, feature in enumerate(feature_names):
            st.write(f"{i + 1}. {feature} ({importance[indices[i]]})")

elif selected_tab == 'Mô hình dự đoán':
    st.subheader("Mô hình dự đoán")

    if 'preprocessed_df' not in st.session_state:
        st.write("Vui lòng thực hiện tiền xử lí dữ liệu trước khi thực hiện Mô hình dự đoán.")
    else:
        preprocessed_df = st.session_state['preprocessed_df']
        st.write('DataFrame đã được tiền xử lí:')
        st.write(preprocessed_df)

        algorithm = st.selectbox("Chọn thuật toán:", ['Phân lớp', 'Hồi quy'])

        if algorithm == 'Phân lớp':
            X = preprocessed_df.iloc[:, :-1]
            y = preprocessed_df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            selected_algorithms = st.multiselect("Chọn thuật toán phân lớp:", ['Logistic Regression', 'Decision Tree',
                                                                               'SVM', 'K-Nearest Neighbors', 'Naive Bayes',
                                                                               'Gradient Boosting'])

            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'SVM': SVC(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            train_accuracies = []
            test_accuracies = []

            for model_name, model in models.items():
                if model_name in selected_algorithms:
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    train_accuracies.append((model_name, train_accuracy))
                    test_accuracies.append((model_name, test_accuracy))

            st.write('Kết quả phân lớp:')
            result_df = pd.DataFrame(test_accuracies, columns=['Model', 'Accuracy'])
            st.write(result_df)

            fig = px.bar(result_df, x='Model', y='Accuracy', title='Độ chính xác của các mô hình phân lớp')
            st.plotly_chart(fig, use_container_width=True)

        elif algorithm == 'Hồi quy':
            X = preprocessed_df.iloc[:, :-1]
            y = preprocessed_df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            selected_algorithms = st.multiselect("Chọn thuật toán hồi quy:", ['Linear Regression', 'Decision Tree',
                                                                              'SVR', 'K-Nearest Neighbors',
                                                                              'Gradient Boosting'])

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'SVR': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }

            train_mses = []
            test_mses = []

            for model_name, model in models.items():
                if model_name in selected_algorithms:
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_mses.append((model_name, train_mse))
                    test_mses.append((model_name, test_mse))

            st.write('Kết quả hồi quy:')
            result_df = pd.DataFrame(test_mses, columns=['Model', 'Mean Squared Error'])
            st.write(result_df)

            fig = px.bar(result_df, x='Model', y='Mean Squared Error', title='Độ chính xác của các mô hình hồi quy')
            st.plotly_chart(fig, use_container_width=True)

elif selected_tab == 'Phân Tích Mô Tả':
    st.subheader("Phân tích dữ liệu")
    if df is not None:
        df_cleaned = df.dropna().reset_index(drop=True)

        numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

        st.write('Thông tin dữ liệu:')
        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write('Thống kê mô tả:')
        st.write(df_cleaned.describe())

        if not numerical_columns.empty:
            st.write('Biểu đồ tương quan:')
            correlation_matrix = df_cleaned[numerical_columns].corr()
            fig, ax = plt.subplots(figsize=(8, 6)) 
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.write('Phân phối các biến số:')
            for column in numerical_columns[:10]:  
                fig, ax = plt.subplots(figsize=(6, 4))  
                sns.histplot(df_cleaned[column], ax=ax, kde=True)
                ax.set_title(f'Phân phối {column}')
                st.pyplot(fig)

        else:
            st.write('Không có biến số nào để phân tích.')

        if not categorical_columns.empty:
            st.write('Phân tích biến phân loại:')
            for column in categorical_columns[:10]:  
                fig, ax = plt.subplots(figsize=(6, 4)) 
                value_counts = df_cleaned[column].value_counts()
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal') 
                ax.set_title(f'Biến {column}')
                st.pyplot(fig)
        else:
            st.write('Không có biến phân loại nào để phân tích.')

    else:
        st.write('Chưa có dữ liệu đầu vào. Vui lòng tải dữ liệu trước khi thực hiện phân tích.')

elif selected_tab == 'Phân tích thống kê':
    st.subheader("Phân tích thống kê")
    if df is not None:
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

        st.write('Thông tin dữ liệu:')
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write('Thống kê mô tả:')
        st.write(df.describe())

        st.write('Biểu đồ tương quan:')
        correlation_matrix = df[numerical_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))  
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write('Phân phối biến số:')
        for column in numerical_columns[:10]:  
            fig, ax = plt.subplots(figsize=(6, 4)) 
            sns.histplot(df[column], ax=ax)
            ax.set_title(f'Phân phối {column}')
            st.pyplot(fig)

        st.write('Giá trị ngoại lai:')
        for column in numerical_columns[:10]: 
            fig, ax = plt.subplots(figsize=(6, 4))  
            sns.boxplot(x=df[column], ax=ax)
            ax.set_title(f'Giá trị ngoại lai của {column}')
            st.pyplot(fig)

        st.write('Phân tích biến phân loại:')
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns[:10]: 
            st.write(f'Biến {column}:')
            st.write(df[column].value_counts())

    else:
        st.write('Chưa có dữ liệu đầu vào. Vui lòng tải dữ liệu trước khi thực hiện phân tích.')



















