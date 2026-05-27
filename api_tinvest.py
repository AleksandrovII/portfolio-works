import os
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from t_tech.invest import Client, PortfolioResponse, PortfolioPosition, LastPrice, InstrumentIdType

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# TOKEN = os.getenv("INVEST_TOKEN")

TOKEN = "t.cY2p_sUdaAU8VFOFaLKw0Gv7hbdISWAit6rbeSsZWQcgjg0eY_ZGKI5mbVu7BJdtQ-QRfhxcgqAhWLJnh2y-bA"

if not TOKEN:
    raise ValueError("Не задана переменная окружения INVEST_TOKEN. Установите ее командой: export INVEST_TOKEN='ваш_токен'")

def get_currency_rate(client: Client, currency_figi: str) -> Optional[Decimal]:
    """
    Получает последнюю цену (курс) для валютной пары по её FIGI.
    Например, для получения курса USD/RUB использует FIGI доллара.
    Возвращает Decimal или None в случае ошибки.
    """
    try:
        response = client.market_data.get_last_prices(figi=[currency_figi])
        if response.last_prices:
            last_price: LastPrice = response.last_prices[0]
            # Конвертация Quotation в Decimal
            price = Decimal(last_price.price.units) + Decimal(last_price.price.nano) / Decimal(1e9)
            logger.info(f"Курс для FIGI {currency_figi}: {price}")
            return price
    except Exception as e:
        logger.error(f"Ошибка при получении курса для FIGI {currency_figi}: {e}")
    return None

def get_instrument_info_by_figi(client: Client, figi: str) -> Optional[Dict]:
    """
    Получает основную информацию об инструменте (акция, облигация, валюта) по его FIGI.
    Использует метод get_assets для поиска, как в предоставленном вами примере.
    """
    try:
        # Используем прямой поиск инструмента по FIGI
        # В t_tech.invest для поиска по FIGI используется метод find_instrument с указанием id_type
        response = client.instruments.find_instrument(query=figi)
        
        for instrument in response.instruments:
            # Проверяем, что нашли именно наш инструмент (FIGI должен совпадать)
            # У некоторых инструментов FIGI может быть в атрибуте 'figi' или 'id'
            instrument_figi = getattr(instrument, 'figi', getattr(instrument, 'id', None))
            if instrument_figi == figi:
                return {
                    "name": getattr(instrument, 'name', 'Неизвестно'),
                    "ticker": getattr(instrument, 'ticker', 'N/A'),
                    "currency": getattr(instrument, 'currency', 'N/A'),
                    "lot": getattr(instrument, 'lot', 1),
                    "instrument_type": getattr(instrument, 'instrument_type', 'N/A')
                }
        
        logger.warning(f"Инструмент с FIGI {figi} не найден в результатах поиска.")
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении информации об инструменте {figi}: {e}")
        return None

def calculate_position_value(pos: PortfolioPosition, currency_rates: Dict[str, Decimal]) -> Dict:
    """
    Рассчитывает детализированную стоимость позиции с учётом валюты и НКД.
    """
    # Получаем валюту позиции
    position_currency = 'rub'  # По умолчанию
    if pos.current_price:
        position_currency = pos.current_price.currency.lower()
    
    # Расчёт стоимости позиции в её валюте
    position_value = Decimal('0')
    if pos.current_price and pos.quantity:
        price = Decimal(pos.current_price.units) + Decimal(pos.current_price.nano) / Decimal(1e9)
        quantity = Decimal(pos.quantity.units) + Decimal(pos.quantity.nano) / Decimal(1e9)
        position_value = price * quantity
    
    # Добавление НКД (важно для облигаций [citation:7])
    nkd_value = Decimal('0')
    if pos.current_nkd:
        nkd = Decimal(pos.current_nkd.units) + Decimal(pos.current_nkd.nano) / Decimal(1e9)
        nkd_value = nkd
        position_value += nkd
    
    # Конвертация в рубли, если необходимо
    rub_value = position_value
    if position_currency != 'rub' and position_value > 0:
        rate = currency_rates.get(position_currency)
        if rate:
            rub_value = position_value * rate
        else:
            logger.warning(f"Курс для валюты '{position_currency}' не найден. Конвертация не выполнена.")
            rub_value = position_value  # Оставляем в исходной валюте
    
    return {
        "position_value": position_value,
        "position_currency": position_currency,
        "rub_value": rub_value,
        "nkd": nkd_value
    }

def get_portfolio_summary(account_id: str, client: Client, currency_rates: Dict[str, Decimal]) -> Optional[Dict]:
    """
    Запрашивает и обрабатывает портфель для конкретного счета.
    """
    try:
        portfolio: PortfolioResponse = client.operations.get_portfolio(account_id=account_id)
    except Exception as e:
        logger.error(f"Ошибка при получении портфеля для счета {account_id}: {e}")
        return None

    summary = {
        "total_amount_portfolio": portfolio.total_amount_portfolio,
        "positions": []
    }

    for pos in portfolio.positions:
        instrument_info = get_instrument_info_by_figi(client, pos.figi) or {
            "name": "Неизвестный инструмент",
            "ticker": pos.figi[:10],  # Берем часть FIGI как тикер
            "currency": "N/A",
            "lot": 1,
            "instrument_type": "N/A"
        }
        
        # Расчёт стоимости позиции
        value_info = calculate_position_value(pos, currency_rates)
        
        position_data = {
            "figi": pos.figi,
            **instrument_info,
            "quantity": pos.quantity,
            "average_position_price": pos.average_position_price,
            "expected_yield": pos.expected_yield,
            "current_nkd": pos.current_nkd,
            "current_price": pos.current_price,
            **value_info
        }
        
        summary["positions"].append(position_data)

    return summary

def get_common_currency_rates(client: Client) -> Dict[str, Decimal]:
    """
    Получает курсы для основных валют относительно рубля.
    FIGI основных валютных пар с Московской биржи:
    """
    currency_figi_map = {
        'usd': 'BBG0013HGFT4',  # USD/RUB
        'eur': 'BBG0013HJJ31',  # EUR/RUB
        'cny': 'BBG0013HRTL0',  # CNY/RUB
        'hkd': 'BBG0013HSW87',  # HKD/RUB
        'chf': 'BBG0013HQ5K4',  # CHF/RUB
    }
    
    rates = {'rub': Decimal('1.0')}  # Рубль к рублю = 1
    
    for curr, figi in currency_figi_map.items():
        rate = get_currency_rate(client, figi)
        if rate:
            rates[curr] = rate
        else:
            logger.warning(f"Не удалось получить курс для {curr.upper()}. Будет использовано значение 1.")
            rates[curr] = Decimal('1.0')
    
    return rates

def main():
    logger.info("Подключение к T-Invest API...")
    with Client(TOKEN) as client:
        try:
            # 1. Получение актуальных курсов валют
            currency_rates = get_common_currency_rates(client)
            logger.info("Курсы валют загружены.")
            
            # 2. Получение списка счетов
            accounts_response = client.users.get_accounts()
            accounts = accounts_response.accounts
            logger.info(f"Найдено счетов: {len(accounts)}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации: {e}")
            return

        if not accounts:
            logger.warning("Не найдено ни одного инвестиционного счета.")
            return

        total_portfolio_value_rub = Decimal('0')
        all_positions = []

        # 3. Обработка каждого счета
        for account in accounts:
            logger.info(f"Обработка счета: {account.name} (ID: {account.id})")
            portfolio_summary = get_portfolio_summary(account.id, client, currency_rates)

            if not portfolio_summary:
                continue
            
            # 4. Суммирование стоимости портфеля в рублях
            account_value_rub = Decimal('0')
            for pos in portfolio_summary["positions"]:
                account_value_rub += pos["rub_value"]
                all_positions.append({
                    "account": account.name,
                    **pos
                })
            
            total_portfolio_value_rub += account_value_rub
            logger.info(f"  Стоимость портфеля на счете: {account_value_rub:.2f} RUB")

        # 5. Вывод итогового отчёта
        print("\n" + "="*100)
        print(f"ОБЩИЙ БАЛАНС ПОРТФЕЛЯ: {total_portfolio_value_rub:.2f} RUB")
        print("="*100)

        if not all_positions:
            print("\nНа счетах нет открытых позиций.")
            return

        print("\nДЕТАЛИЗАЦИЯ ПО ПОЗИЦИЯМ:")
        print("-"*100)
        header = f"{'Счет':<15} {'Тикер':<10} {'Название':<25} {'Валюта':<6} {'Кол-во':>10} {'Ср. цена':>12} {'Тек. цена':>12} {'Стоимость':>14} {'НКД':>10}"
        print(header)
        print("-"*100)

        for pos in all_positions:
            # Форматирование количества
            qty_str = "0"
            if pos["quantity"]:
                qty = Decimal(pos["quantity"].units) + Decimal(pos["quantity"].nano) / Decimal(1e9)
                qty_str = f"{qty:.2f}"
            
            # Форматирование средней цены
            avg_price_str = "N/A"
            if pos["average_position_price"]:
                avg = Decimal(pos["average_position_price"].units) + Decimal(pos["average_position_price"].nano) / Decimal(1e9)
                currency = pos["average_position_price"].currency.lower()
                avg_price_str = f"{avg:.2f} {currency}"
            
            # Форматирование текущей цены
            curr_price_str = "N/A"
            if pos["current_price"]:
                curr = Decimal(pos["current_price"].units) + Decimal(pos["current_price"].nano) / Decimal(1e9)
                currency = pos["current_price"].currency.lower()
                curr_price_str = f"{curr:.2f} {currency}"
            
            # Стоимость с указанием исходной валюты
            if pos["position_currency"] != 'rub':
                value_str = f"{pos['position_value']:.2f} {pos['position_currency'].upper()} ({pos['rub_value']:.2f} RUB)"
            else:
                value_str = f"{pos['rub_value']:.2f} RUB"
            
            # Форматирование НКД
            nkd_str = f"{pos['nkd']:.2f}" if pos['nkd'] else "0.00"
            
            # Вывод строки
            row = f"{pos['account'][:14]:<15} {pos['ticker'][:9]:<10} {pos['name'][:24]:<25} {pos['position_currency'].upper():<6} {qty_str:>10} {avg_price_str:>12} {curr_price_str:>12} {value_str:>14} {nkd_str:>10}"
            print(row)
        
        print("-"*100)
        
        # 6. Аналитика по валютам
        print("\n РАСПРЕДЕЛЕНИЕ СТОИМОСТИ ПО ВАЛЮТАМ:")
        currency_summary = {}
        for pos in all_positions:
            curr = pos["position_currency"].upper()
            currency_summary[curr] = currency_summary.get(curr, Decimal('0')) + pos["position_value"]
        
        for curr, value in sorted(currency_summary.items()):
            if curr != 'RUB':
                rub_value = value * currency_rates.get(curr.lower(), Decimal('1'))
                print(f"  {curr}: {value:.2f} ~ {rub_value:.2f} RUB")
            else:
                print(f"  {curr}: {value:.2f}")

if __name__ == "__main__":
    main()