# RnD

Команда: !False progers

Прогнозируется направление изменения курса валютной пары USD/RUB на один день вперед на основе изменения цен индексов за вчерашний день и движения цены доллара за прошедшие периоды. Для модели использовали временные ряды и классификатор. Регрессионные модели показали себя плохо, ошибка была больше, чем изменение цены валюты за день, модель не имела прогнозной ценности, алгоритмы классификации показали себя лучше. Для улучшения модели можно разжиться большим количеством данных для обучения, изменить представление временных рядов, попробовать прогнозировать на 3-5 дней вперед и попробовать кластеризацию.

### Установка

1. Пререквизиты: docker (20.10.14), docker-compose (1.29.2)

2. Запуск сервиса
```
docker-compose up --build -d
```

3. API будет доступно на `localhost:8080`

### Endpoints

**/predict**

Предсказывает рост или падения курса USD/RUB

Body
```
{
    "DATE": "2021-05-07",
    "OPEN_cur": 61.1850,
    "HIGH_cur": 61.1975,
    "LOW_cur": 61.1850,
    "CLOSE_cur": 61.1970,
    "VOL_cur": 36311.0
}
```

- `DATE` (string "%Y-%m-%d") --- дата, относительно которой предсказывается курс следующего дня
- `OPEN_cur` (float) --- цена на открытии биржи
- `HIGH_cur` (float) --- наибольшая цена за день
- `LOW_cur` (float) --- наименьшаа цена за день
- `CLOSE_cur` (float) --- цена на закрытии биржи
- `VOL_cur` (float) --- количество сделок


Ответ: 0 -- падение курса, 1 -- рост курса


### Ссылки

[Backend](https://github.com/tutkarma/scb-nfp-backend)

[Frontend]()