# FoodDelhivery

import os
import sqlite3
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
 
random.seed(42)
np.random.seed(42)
 
DB_FILE = "foodflow.db"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)
 
 
# ------------------------------------------------------------------
# 1. DATABASE LAYER
# ------------------------------------------------------------------
def create_tables(conn):
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS menu;
        DROP TABLE IF EXISTS customers;
 
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            area        TEXT NOT NULL
        );
        CREATE TABLE menu (
            item_id   INTEGER PRIMARY KEY,
            item_name TEXT NOT NULL,
            category  TEXT NOT NULL,
            price     REAL NOT NULL
        );
        CREATE TABLE orders (
            order_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   INTEGER,
            item_id       INTEGER,
            quantity      INTEGER,
            order_date    TEXT,
            order_hour    INTEGER,
            distance_km   REAL,
            delivery_min  REAL,
            status        TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (item_id)     REFERENCES menu(item_id)
        );
    """)
    conn.commit()
 
 
MENU = [
    (1, "Veg Burger",     "Fast Food", 120),
    (2, "Paneer Pizza",   "Fast Food", 280),
    (3, "Veg Biryani",    "Main Course", 180),
    (4, "Butter Naan Combo", "Main Course", 150),
    (5, "Masala Dosa",    "South Indian", 110),
    (6, "Chole Bhature",  "North Indian", 130),
    (7, "Pasta Alfredo",  "Italian", 220),
    (8, "Cold Coffee",    "Beverages", 90),
]
AREAS = ["Model Town", "Civil Lines", "Adarsh Nagar", "Basti Sheikh", "Rama Mandi"]
FIRST = ["Aman", "Simran", "Rohit", "Neha", "Gurpreet", "Pooja", "Karan", "Jasleen",
         "Vikram", "Anjali", "Harpreet", "Riya", "Manav", "Kiran", "Sahil",
         "Tanvi", "Arjun", "Meera", "Deepak", "Isha"]
 
 
def add_customer(conn, cid, name, area):                 # CREATE
    conn.execute("INSERT INTO customers VALUES (?,?,?)", (cid, name, area))
 
 
def add_order(conn, cid, item, qty, date, hour, dist, mins, status):   # CREATE
    conn.execute("""INSERT INTO orders (customer_id,item_id,quantity,order_date,
                    order_hour,distance_km,delivery_min,status)
                    VALUES (?,?,?,?,?,?,?,?)""",
                 (cid, item, qty, date, hour, dist, mins, status))
 
 
def update_status(conn, order_id, new_status):           # UPDATE
    conn.execute("UPDATE orders SET status=? WHERE order_id=?", (new_status, order_id))
 
 
def delete_order(conn, order_id):                        # DELETE
    conn.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
 
 
# ------------------------------------------------------------------
# 2. SAMPLE DATA GENERATOR (simulated data for learning and testing)
# ------------------------------------------------------------------
def populate(conn, n_orders=600):
    conn.executemany("INSERT INTO menu VALUES (?,?,?,?)", MENU)
    for i, name in enumerate(FIRST, start=1):
        add_customer(conn, i, name, random.choice(AREAS))
 
    dates = pd.date_range("2026-01-01", "2026-06-30")
    # some items are more popular than others
    weights = [0.18, 0.12, 0.16, 0.08, 0.14, 0.12, 0.08, 0.12]
    for _ in range(n_orders):
        cid = random.randint(1, len(FIRST))
        item = random.choices(range(1, 9), weights)[0]
        qty = random.choices([1, 2, 3, 4], [0.45, 0.3, 0.15, 0.1])[0]
        date = random.choice(dates)
        hour = random.choices([9, 12, 13, 14, 17, 19, 20, 21, 22],
                              [3, 14, 16, 8, 6, 14, 18, 14, 7])[0]
        dist = round(random.uniform(0.5, 9.0), 1)
        peak = 1 if hour in (13, 20, 21) else 0
        weekend = 1 if date.weekday() >= 5 else 0
        mins = 12 + 3.2 * dist + 6 * peak + 3 * weekend + 1.5 * qty + np.random.normal(0, 3)
        mins = round(max(10, mins), 1)
        add_order(conn, cid, item, qty, date.strftime("%Y-%m-%d"), hour, dist, mins, "Delivered")
    conn.commit()
 
 
def load_data(conn):
    q = """SELECT o.order_id, o.customer_id, c.name, c.area, m.item_name, m.category,
                   m.price, o.quantity, o.order_date, o.order_hour, o.distance_km,
                   o.delivery_min, o.status
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN menu m      ON o.item_id = m.item_id"""
    df = pd.read_sql_query(q, conn)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["month"] = df["order_date"].dt.month_name()
    df["month_no"] = df["order_date"].dt.month
    df["weekday_no"] = df["order_date"].dt.weekday
    df["is_weekend"] = (df["weekday_no"] >= 5).astype(int)
    df["is_peak"] = df["order_hour"].isin([13, 20, 21]).astype(int)
    return df
 
 
def heading(text):
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)
 
 
def main():
    conn = sqlite3.connect(DB_FILE)
 
    # ---------------- Database demo ----------------
    heading("STEP 1: DATABASE SETUP AND CRUD OPERATIONS")
    create_tables(conn)
    populate(conn)
    total = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    print("Tables created: customers, menu, orders")
    print("Records inserted in orders table:", total)
 
    add_order(conn, 3, 1, 2, "2026-07-01", 20, 3.5, 30.0, "Preparing")   # CREATE
    new_id = conn.execute("SELECT MAX(order_id) FROM orders").fetchone()[0]
    print("CREATE : new order added with order_id =", new_id)
    update_status(conn, new_id, "Out for Delivery")                       # UPDATE
    row = conn.execute("SELECT order_id,status FROM orders WHERE order_id=?",
                       (new_id,)).fetchone()
    print("READ   :", row)
    delete_order(conn, new_id)                                            # DELETE
    print("DELETE : order", new_id, "removed. Total orders now =",
          conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
    conn.commit()
 
    # ---------------- Data loading ----------------
    df = load_data(conn)
    heading("STEP 2: DATASET OVERVIEW (PANDAS)")
    print("Shape of dataset (rows, columns):", df.shape)
    print("\nFirst 5 records:")
    print(df[["order_id", "name", "item_name", "quantity", "order_date",
              "distance_km", "delivery_min", "revenue"]].head().to_string(index=False))
    print("\nMissing values in dataset:", int(df.isnull().sum().sum()))
    print("\nSummary statistics:")
    print(df[["quantity", "distance_km", "delivery_min", "revenue"]].describe().round(2))
 
    # ---------------- Analytics ----------------
    heading("STEP 3: DATA ANALYTICS")
    print("Total revenue            : Rs.", int(df["revenue"].sum()))
    print("Average order value      : Rs.", round(df["revenue"].mean(), 2))
    print("Average delivery time    :", round(df["delivery_min"].mean(), 1), "minutes")
 
    popular = df.groupby("item_name")["quantity"].sum().sort_values(ascending=False)
    print("\nTop 5 popular food items (units sold):")
    print(popular.head(5).to_string())
 
    month_order = ["January", "February", "March", "April", "May", "June"]
    monthly = df.groupby("month").agg(orders=("order_id", "count"),
                                      revenue=("revenue", "sum"),
                                      avg_delivery=("delivery_min", "mean")
                                      ).reindex(month_order).round(1)
    print("\nMonthly order report:")
    print(monthly.to_string())
 
    area = df.groupby("area")["delivery_min"].mean().round(1).sort_values()
    print("\nAverage delivery time by area (minutes):")
    print(area.to_string())
 
    # ---------------- Visualisation ----------------
    heading("STEP 4: DATA VISUALISATION")
    sns.set_theme(style="whitegrid")
    navy, orange = "#14213D", "#F77F00"
 
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=popular.values, y=popular.index, color=orange, ax=ax)
    ax.set_title("Popular Food Items (Units Sold)", color=navy, fontweight="bold")
    ax.set_xlabel("Units sold"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/popular_items.png", dpi=150); plt.close()
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly.index, monthly["revenue"], marker="o", color=orange, linewidth=2.5)
    ax.set_title("Monthly Revenue (Rs.)", color=navy, fontweight="bold")
    ax.set_ylabel("Revenue"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/monthly_revenue.png", dpi=150); plt.close()
 
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x="distance_km", y="delivery_min", hue="is_peak",
                    palette=[navy, orange], alpha=0.7, ax=ax)
    ax.set_title("Distance vs Delivery Time", color=navy, fontweight="bold")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery time (min)")
    ax.legend(title="Peak hour")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/distance_vs_time.png", dpi=150); plt.close()
 
    fig, ax = plt.subplots(figsize=(7, 4))
    hourly = df.groupby("order_hour")["order_id"].count()
    sns.barplot(x=hourly.index, y=hourly.values, color=navy, ax=ax)
    ax.set_title("Orders by Hour of Day", color=navy, fontweight="bold")
    ax.set_xlabel("Hour (24h)"); ax.set_ylabel("Orders")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/orders_by_hour.png", dpi=150); plt.close()
    print("Charts saved in folder:", OUT_DIR)
    print(sorted(os.listdir(OUT_DIR)))
 
    # ---------------- AI / ML ----------------
    heading("STEP 5A: AI - DELIVERY TIME PREDICTION (REGRESSION)")
    feats = ["distance_km", "quantity", "order_hour", "is_peak", "is_weekend"]
    X, y = df[feats], df["delivery_min"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
    for name, model in [("Linear Regression", lr), ("Random Forest", rf)]:
        p = model.predict(Xte)
        print(f"{name:18s} MAE = {mean_absolute_error(yte, p):5.2f} min | R2 = {r2_score(yte, p):.3f}")
    sample = pd.DataFrame([[4.0, 2, 20, 1, 0]], columns=feats)
    print("\nPrediction: 4 km, 2 items, 8 PM, peak hour, weekday ->",
          round(float(lr.predict(sample)[0]), 1), "minutes")
 
    heading("STEP 5B: AI - DELIVERY DELAY DETECTION (CLASSIFICATION)")
    df["is_delayed"] = (df["delivery_min"] > 40).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(df[feats], df["is_delayed"],
                                          test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print("Rule used: delivery time > 40 minutes = Delayed")
    print("Delayed orders in dataset:", int(df["is_delayed"].sum()),
          f"({df['is_delayed'].mean()*100:.1f}%)")
    print("Model accuracy on test data:", round(acc * 100, 1), "%")
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nFeature importance:")
    print(imp.round(3).to_string())
 
    heading("STEP 5C: AI - FOOD DEMAND FORECASTING")
    daily = df.groupby("order_date")["order_id"].count().reset_index(name="orders")
    daily["day_no"] = np.arange(len(daily))
    daily["weekday_no"] = daily["order_date"].dt.weekday
    Xd, yd = daily[["day_no", "weekday_no"]], daily["orders"]
    split = int(len(daily) * 0.85)
    dm = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xd[:split], yd[:split])
    pred = dm.predict(Xd[split:])
    print("Training days:", split, "| Testing days:", len(daily) - split)
    print("MAE (orders per day):", round(mean_absolute_error(yd[split:], pred), 2))
    future = pd.DataFrame({"day_no": range(len(daily), len(daily) + 7),
                           "weekday_no": [(daily["weekday_no"].iloc[-1] + i + 1) % 7 for i in range(7)]})
    fc = dm.predict(future)
    print("Forecast of orders for next 7 days:", [round(float(v), 1) for v in fc])
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(daily["order_date"][split:], yd[split:], label="Actual", color=navy, marker="o")
    ax.plot(daily["order_date"][split:], pred, label="Predicted", color=orange, marker="s")
    ax.set_title("Demand Forecast: Actual vs Predicted", color=navy, fontweight="bold")
    ax.legend(); plt.xticks(rotation=25)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/demand_forecast.png", dpi=150); plt.close()
 
    heading("STEP 5D: AI - CUSTOMER SEGMENTATION (K-MEANS)")
    cust = df.groupby("name").agg(orders=("order_id", "count"),
                                  spend=("revenue", "sum")).reset_index()
    sc = StandardScaler().fit_transform(cust[["orders", "spend"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sc)
    cust["cluster"] = km.labels_
    order = cust.groupby("cluster")["spend"].mean().sort_values().index.tolist()
    names = {order[0]: "Occasional", order[1]: "Regular", order[2]: "Loyal (High value)"}
    cust["segment"] = cust["cluster"].map(names)
    print(cust.groupby("segment").agg(customers=("name", "count"),
                                      avg_orders=("orders", "mean"),
                                      avg_spend=("spend", "mean")).round(1).to_string())
 
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=cust, x="orders", y="spend", hue="segment",
                    palette=["#8D99AE", navy, orange], s=110, ax=ax)
    ax.set_title("Customer Segments (K-Means)", color=navy, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/customer_segments.png", dpi=150); plt.close()
 
    heading("STEP 5E: AI - REVENUE PREDICTION (DAILY REVENUE)")
    drev = df.groupby("order_date")["revenue"].sum().reset_index()
    drev["day_no"] = np.arange(len(drev))
    drev["weekday_no"] = drev["order_date"].dt.weekday
    drev["is_weekend"] = (drev["weekday_no"] >= 5).astype(int)
    cols = ["day_no", "weekday_no", "is_weekend"]
    sp = int(len(drev) * 0.85)
    rm = RandomForestRegressor(n_estimators=100, random_state=42).fit(drev[cols][:sp], drev["revenue"][:sp])
    pr = rm.predict(drev[cols][sp:])
    base = np.full(len(pr), drev["revenue"][:sp].mean())      # simple baseline
    print("Model MAE    (Rs. per day):", round(mean_absolute_error(drev["revenue"][sp:], pr), 1))
    print("Baseline MAE (Rs. per day):", round(mean_absolute_error(drev["revenue"][sp:], base), 1))
    fut = pd.DataFrame({"day_no": range(len(drev), len(drev) + 7)})
    fut["weekday_no"] = [(drev["weekday_no"].iloc[-1] + i + 1) % 7 for i in range(7)]
    fut["is_weekend"] = (fut["weekday_no"] >= 5).astype(int)
    print("Predicted revenue for next 7 days: Rs.", int(rm.predict(fut[cols]).sum()))
    print("Note: sample data is simulated, so accuracy will improve with real orders.")
 
    heading("FOODFLOW PROTOTYPE RUN COMPLETED")
    print("Status: Prototype working on sample data. Streamlit dashboard is planned.")
    conn.close()
 
 
if __name__ == "__main__":
    main()
 

 
