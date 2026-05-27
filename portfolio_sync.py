import csv
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from scipy.optimize import minimize
from t_tech.invest import AsyncClient, Client, CandleInterval, OrderExecutionReportStatus
from t_tech.invest.utils import now
import functools
from t_tech.invest import OperationState
from time import sleep


def get_currency_rate(client, figi):
    try:
        response = client.market_data.get_last_prices(figi=[figi])
        if response.last_prices:
            price = response.last_prices[0].price
            return Decimal(price.units) + Decimal(price.nano) / Decimal(1e9)
    except:
        return Decimal('1')

def get_instrument_info(client, figi):
    try:
        response = client.instruments.get_instrument_by(id_type=1, id=figi)
        instr = response.instrument
        instrument_type = instr.instrument_type
        
        # Проверяем, является ли это золотом
        name_lower = instr.name.lower()
        ticker_lower = instr.ticker.lower()
        
        # Если это золото или драгметалл, меняем тип
        if ('золот' in name_lower or 'gold' in name_lower or 
            'gld' in ticker_lower or 'metal' in name_lower):
            instrument_type = 'precious_metal'
        
        return {
            "name": instr.name,
            "ticker": instr.ticker,
            "currency": instr.currency,
            "instrument_type": instrument_type
        }
    except:
        # Проверяем по FIGI или названию
        if 'GLD' in figi or 'GOLD' in figi:
            return {"name": "Золото", "ticker": "GLD", "currency": "RUB", "instrument_type": "precious_metal"}
        return {"name": "", "ticker": figi[:10], "currency": "N/A", "instrument_type": "N/A"}

def input_date(prompt):
    while True:
        s = input(prompt)
        try:
            d = datetime.strptime(s, "%Y-%m-%d")
            return d.replace(tzinfo=timezone.utc)
        except ValueError:
            print("Неверный формат. Используйте ГГГГ-ММ-ДД")

def calculate_position_value(pos, rates, instrument_currency):
    currency = instrument_currency.lower() if instrument_currency != 'N/A' else 'rub'
    
    if pos.current_price and hasattr(pos.current_price, 'currency'):
        currency = pos.current_price.currency.lower()
    
    quantity = Decimal(pos.quantity.units) + Decimal(pos.quantity.nano) / Decimal(1e9) if pos.quantity else Decimal('0')
    
    price = Decimal('0')
    if pos.current_price:
        price = Decimal(pos.current_price.units) + Decimal(pos.current_price.nano) / Decimal(1e9)
    
    value_in_currency = price * quantity if price > 0 else Decimal('0')
    
    nkd = Decimal('0')
    if pos.current_nkd:
        nkd = Decimal(pos.current_nkd.units) + Decimal(pos.current_nkd.nano) / Decimal(1e9)
        value_in_currency += nkd
    
    if currency != 'rub':
        rate = rates.get(currency, Decimal('1'))
        rub_value = value_in_currency * rate
    else:
        rub_value = value_in_currency
    
    return {
        "value_in_currency": value_in_currency,
        "currency": currency,
        "rub_value": rub_value,
        "nkd": nkd
    }

def get_portfolio(client, account_id, rates):
    try:
        portfolio = client.operations.get_portfolio(account_id=account_id)
        positions = []
        
        for pos in portfolio.positions:
            info = get_instrument_info(client, pos.figi)
            value = calculate_position_value(pos, rates, info["currency"])
            
            current_price = None
            if pos.current_price:
                price_units = pos.current_price.units
                price_nano = pos.current_price.nano
                current_price = Decimal(price_units) + Decimal(price_nano) / Decimal(1e9)
            
            positions.append({
                "figi": pos.figi,
                "account": account_id[:8],
                **info,
                "quantity": pos.quantity,
                "current_price": current_price,
                "average_position_price": pos.average_position_price,
                **value
            })
        
        return positions
    except Exception as e:
        print(f"Ошибка: {e}")
        return []


@functools.lru_cache(maxsize=256)
def get_instrument_info_cached(client, figi, retries=3):
    """Получить информацию об инструменте с кэшированием и повторными попытками при RESOURCE_EXHAUSTED"""
    for attempt in range(retries):
        try:
            response = client.instruments.get_instrument_by(id_type=1, id=figi)
            instr = response.instrument
            return {
                "ticker": instr.ticker,
                "name": instr.name,
                "type": instr.instrument_type
            }
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < retries - 1:
                sleep(1 * (attempt + 1))  # увеличиваем паузу
                continue
            # Другая ошибка или последняя попытка
            break
    # Возвращаем заглушку, если не удалось
    return {"ticker": figi[:10], "name": "N/A", "type": "N/A"}


def save_to_csv(positions, orders, filename="portfolio_report.xlsx"):
    if positions:
        portfolio_data = []
        for p in positions:
            qty = Decimal(p['quantity'].units) + Decimal(p['quantity'].nano)/Decimal(1e9) if p['quantity'] else 0
            portfolio_data.append({
                'Счет': p['account'],
                'Тикер': p['ticker'],
                'Название': p['name'][:30],
                'Тип': p['instrument_type'],
                'Валюта': p['currency'].upper(),
                'Количество': float(qty),
                'Текущая цена': float(p['current_price']) if p['current_price'] else 0,
                'Стоимость в валюте': float(p['value_in_currency']),
                'Стоимость в RUB': float(p['rub_value']),
                'FIGI': p['figi']
            })
        
        orders_data = []
        for o in orders:
            orders_data.append({
                'Дата': o['Дата'],
                'Тикер': o['Тикер'],
                'Направление': o['Направление'],
                'Количество': o['Количество'],
                'Цена': float(Decimal(o['Цена'].replace(' RUB', ''))),
                'Сумма': float(Decimal(o['Сумма'].replace(' RUB', '')))
            })
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pd.DataFrame(portfolio_data).to_excel(writer, sheet_name='Портфель', index=False)
            if orders_data:
                pd.DataFrame(orders_data).to_excel(writer, sheet_name='Сделки', index=False)

async def get_orders(account_id):
    async with AsyncClient(TOKEN) as client:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            orders_response = await client.orders.get_orders(
                account_id=account_id,
                from_=start, to=end,
                execution_status=[
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL
                ]
            )
            
            result = []
            for o in orders_response.orders:
                try:
                    instr = await client.instruments.get_instrument_by(id_type=1, id=o.figi)
                    ticker = instr.instrument.ticker
                except:
                    ticker = o.figi[:10]
                
                price = Decimal(o.executed_order_price.units) + Decimal(o.executed_order_price.nano)/Decimal(1e9) if o.executed_order_price else Decimal('0')
                total = Decimal(o.total_order_amount.units) + Decimal(o.total_order_amount.nano)/Decimal(1e9) if o.total_order_amount else price * o.lots_executed
                
                result.append({
                    'Дата': o.date.strftime('%Y-%m-%d %H:%M'),
                    'Тикер': ticker,
                    'Направление': 'Покупка' if o.direction.value == 1 else 'Продажа',
                    'Количество': o.lots_executed,
                    'Цена': f"{price:.2f} RUB",
                    'Сумма': f"{total:.2f} RUB"
                })
            return result
        except:
            return []

def plot_assets(positions, total_value):
    if not positions:
        return
    
    # Группировка с учетом драгоценных металлов
    asset_types = {}
    for p in positions:
        asset_type = p['instrument_type']
        
        # Определяем тип для отображения
        if asset_type == 'currency':
            asset_type = 'Деньги'
        elif asset_type == 'precious_metal':
            asset_type = 'Драг. металлы'
        elif asset_type == 'share':
            asset_type = 'Акции'
        elif asset_type == 'bond':
            asset_type = 'Облигации'
        elif asset_type == 'etf':
            asset_type = 'Фонды'
        else:
            asset_type = 'Другие'
        
        asset_types[asset_type] = asset_types.get(asset_type, 0) + float(p['rub_value'])
    
    plt.figure(figsize=(12, 5))
    
    # 1. По типам
    plt.subplot(1, 2, 1)
    labels = list(asset_types.keys())
    sizes = list(asset_types.values())
    colors = sns.color_palette("Set2", len(labels))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'Типы активов\n{float(total_value):,.0f} RUB')
    
    # 2. Top-10 активов (исключаем только деньги)
    plt.subplot(1, 2, 2)
    assets = {}
    for p in positions:
        if p['instrument_type'] == 'currency':  # Исключаем только деньги
            continue
        ticker = p['ticker']
        display_name = p['name'][:15] if len(p['name']) > 15 else p['name']
        assets[f"{ticker}\n{display_name}"] = float(p['rub_value'])
    
    top_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_assets:
        tickers = [k for k, _ in top_assets]
        values = [v for _, v in top_assets]
        
        colors = sns.color_palette("viridis", len(tickers))
        bars = plt.barh(range(len(tickers)), values, color=colors)
        plt.yticks(range(len(tickers)), tickers)
        plt.xlabel('RUB')
        plt.title('Топ-10 активов')
        plt.gca().invert_yaxis()
        
        # Добавляем значения на график
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:,.0f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("assets.png", dpi=150, bbox_inches='tight')
    plt.show()

async def get_history(client, figi, days=180):
    try:
        end = now()
        start = end - timedelta(days=days)
        candles = await client.market_data.get_candles(
            instrument_id=figi,
            from_=start, to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY
        )
        prices = []
        for c in candles.candles:
            price = Decimal(c.close.units) + Decimal(c.close.nano)/Decimal(1e9)
            prices.append(float(price))
        return pd.Series(prices) if prices else None
    except:
        return None

async def get_all_history(figi_list):
    async with AsyncClient(TOKEN) as client:
        tasks = [get_history(client, figi) for figi in figi_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {figi: res for figi, res in zip(figi_list, results) 
                if not isinstance(res, Exception) and res is not None}

def markowitz_analysis(positions):
    # Только акции и фонды для анализа Марковица
    analysis_pos = [p for p in positions if p['instrument_type'] in ['share', 'etf']]
    if len(analysis_pos) < 2:
        print("\n⚠ Недостаточно данных для анализа Марковица")
        return
    
    print("\nЗагрузка исторических данных...")
    figis = list(set(p['figi'] for p in analysis_pos))
    history = asyncio.run(get_all_history(figis))
    
    # Фильтруем активы с данными
    valid_data = {}
    for p in analysis_pos:
        if p['figi'] in history and history[p['figi']] is not None and len(history[p['figi']]) > 10:
            valid_data[p['ticker']] = history[p['figi']]
    
    if len(valid_data) < 2:
        print("Недостаточно исторических данных")
        return
    
    returns = pd.DataFrame(valid_data).pct_change().dropna()
    
    # Корреляционная матрица
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Корреляция активов')
    plt.tight_layout()
    plt.savefig("correlation.png", dpi=150)
    plt.show()
    
    # Оптимизация Марковица
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    def portfolio_stats(weights):
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol
    
    n = len(mean_returns)
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    init_weights = np.array([1/n] * n)
    
    opt = minimize(lambda w: -portfolio_stats(w)[0]/portfolio_stats(w)[1] if portfolio_stats(w)[1] > 0 else 0,
                   init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights = opt.x
    opt_return, opt_vol = portfolio_stats(opt_weights)
    
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette('viridis', n)
    plt.bar(range(n), opt_weights * 100, color=colors)
    plt.xticks(range(n), list(valid_data.keys()), rotation=45)
    plt.ylabel('Вес (%)')
    plt.title(f'Оптимальные веса\nДоходность: {opt_return*100:.1f}%, Волатильность: {opt_vol*100:.1f}%')
    plt.tight_layout()
    plt.savefig("optimization.png", dpi=150)
    plt.show()

def main():
    print("Анализ портфеля")
    print("=" * 40)
    
    # Ввод диапазона дат
    print("Введите диапазон дат для анализа покупок")
    from_date = input_date("Начальная дата (ГГГГ-ММ-ДД): ")
    to_date = input_date("Конечная дата (ГГГГ-ММ-ДД): ")
    to_date = to_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    with Client(TOKEN) as client:
        # Курсы валют (если нужны для других частей программы)
        rates = {
            'rub': Decimal('1'),
            'usd': get_currency_rate(client, 'BBG0013HGFT4'),
            'eur': get_currency_rate(client, 'BBG0013HJJ31'),
            'cny': get_currency_rate(client, 'BBG0013HRTL0'),
        }
        
        accounts = client.users.get_accounts().accounts
        print(f"Счетов: {len(accounts)}")
     
        all_positions = []   # для портфеля
        all_orders = []      # для сделок (если нужно)
        all_purchases = []   # сюда собираем покупки

        for acc in accounts:
            print(f"\nСчет: {acc.name}")
            operations = client.operations.get_operations(
                account_id=acc.id,
                from_=from_date,
                to=to_date
            ).operations

            replenishments = [op for op in operations 
                            if op.type and "Пополнение" in op.type]
            print(f"  Найдено пополнений: {len(replenishments)}")

            stock_purchases = []
            for op in operations:
                # 1. Проверяем, что операция исполнена
                if op.state != OperationState.OPERATION_STATE_EXECUTED:
                    continue

                # 2. Приводим тип к нижнему регистру для проверки
                type_lower = op.type.lower() if op.type else ''

                # 3. Ищем ключевые слова покупки, исключая пополнения
                if ('покупка' in type_lower or 'buy' in type_lower) and 'пополнение' not in type_lower:
                    # Получаем информацию об инструменте (с кэшированием)
                    instrument_info = {"ticker": "N/A", "name": "N/A", "type": "N/A"}
                    if hasattr(op, 'figi') and op.figi:
                        instrument_info = get_instrument_info_cached(client, op.figi)

                    # Сумма операции (берём модуль, чтобы избавиться от минуса)
                    amount = Decimal('0')
                    if op.payment:
                        amount = abs(Decimal(op.payment.units) + Decimal(op.payment.nano) / Decimal(1e9))

                    # Количество
                    quantity = 0
                    if hasattr(op, 'quantity') and op.quantity:
                        quantity = op.quantity
                    elif hasattr(op, 'quantity_executed') and op.quantity_executed:
                        quantity = op.quantity_executed

                    # Цена за единицу
                    price = Decimal('0')
                    if hasattr(op, 'price') and op.price:
                        price = Decimal(op.price.units) + Decimal(op.price.nano) / Decimal(1e9)
                    elif quantity > 0:
                        price = amount / Decimal(quantity)

                    purchase_data = {
                        "date": op.date,
                        "ticker": instrument_info["ticker"],
                        "quantity": quantity,
                        "price": price,
                        "amount": amount,
                    }
                    stock_purchases.append(purchase_data)
                    all_purchases.append(purchase_data)

            if stock_purchases:
                print(f"  Найдено покупок: {len(stock_purchases)}")

        all_purchases.sort(key=lambda x: x['date'], reverse = True)

        # Запись в текстовый файл
        if all_purchases:
            with open("purchases.txt", "w", encoding="utf-8") as f:
                f.write("Дата\tТикер\tКоличество\tЦена\tСумма (RUB)\n")
                for p in all_purchases:
                    date_str = p['date'].strftime("%Y-%m-%d")
                    ticker = p['ticker']
                    qty = p['quantity']
                    price = float(p['price'])
                    amount = float(p['amount'])
                    f.write(f"{date_str}\t{ticker}\t{qty}\t{price:.2f}\t{amount:.2f}\n")
            print(f"\nДанные о покупках сохранены в файл purchases.txt ({len(all_purchases)} записей)")
        else:
            print("\nПокупок за указанный период не найдено.")
if __name__ == "__main__":
    main()


            # print(client.operations.get_operations(account_id=acc.id, 
            #                                             from_= datetime(2025, 7, 15, 9, 15, 12, 610000),
            #                                             to = datetime(2026, 1, 29, 9, 15, 12, 610000))[type =='Пополнение брокерского счёта'],
            #                                             )
        #     orders = asyncio.run(get_orders(acc.id))
        #     all_orders.extend(orders)
            
        #     total = sum(p['rub_value'] for p in positions)
        #     print(f"  Позиций: {len(positions)}")
        #     print(f"  Стоимость: {total:,.0f} RUB")
        #     print(f"  Сделок: {len(orders)}")
        
        # if not all_positions:
        #     print("Нет позиций")
        #     return
        
        # # Статистика по типам
        # print("\n" + "="*40)
        # type_stats = {}
        # for p in all_positions:
        #     t = p['instrument_type']
        #     type_stats[t] = type_stats.get(t, 0) + float(p['rub_value'])
        
        # total_value = sum(p['rub_value'] for p in all_positions)
        # print(f"Итого: {total_value:,.0f} RUB")
        
        # print("\nРаспределение:")
        # for t, v in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
        #     pct = (v / float(total_value)) * 100
        #     print(f"  {t}: {v:,.0f} RUB ({pct:.1f}%)")
        
        # # Валюта
        # print("\nКурсы валют:")
        # for currency, rate in rates.items():
        #     if currency != 'rub':
        #         print(f"  1 {currency.upper()} = {rate:.2f} RUB")
        
        # # Сохранение
        # save_to_csv(all_positions, all_orders)
        # print("\n✓ Сохранено в portfolio_report.xlsx")
        
        # # Графики
        # plot_assets(all_positions, total_value)
        # markowitz_analysis(all_positions)

# if __name__ == "__main__":
#     main()