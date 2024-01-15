# yet_another_mlops_project

Из пройденных иструментов автоматизации процесса машинного обучения был выбрал MLFlow. С его помощью решались две задачи:
  1. Поиск лучшего способа получения эмбеддингов - с помощью трансформера или TF-IDF, в сочетании с XGBoost для задачи классификации текста;
  2. Подбор лучших гиперпараметров для соответствующих моделей.

Для этого было создано виртуальное окружение и скрипты, скачан и установлен MLFlow:

![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/57e8bd12-eb15-4bc7-b7fc-c7b6f90e6b68)

Рассмотрим подробнее скрипты проекта:
1. Сначала отбираются нужные колонки в данных:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/801ed166-bae2-4b26-a471-b5ab9921f707)

2. Затем выполнятеся препроцессинг:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/25858b5a-94f1-4a39-9f07-8bf3e0509718)

  С помощью функции из:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/a55796a0-17f1-428b-ace5-2c98186f6568)

  Полученный вид данных будет использоватся при получении эмбедингов с помощью трансформера.
  
3. Эти же данные дополнительно лемматизируем и удалим часть наиболее общих слов:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/c3b8bca5-3531-4854-8eaa-0a909ed5779b)

  С помощью функций из:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/970f719b-4b24-45e7-bc5c-a00cfe106a8d)  

4. Дальше делается train-test split:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/413d9ee2-f182-497b-9f67-5d9d375eac6a)

6. После этого скачаем трансформер и создадим эмбеддинги с его помощью:

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/106f846f-e76b-4455-afd1-d649f00692db)

  ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/52625db5-a3c1-4e93-a3d6-c2a682cf160b)

6. Здесь получим эмбеддинги с помощью TF-IDF векторайзера:

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/504a2b8b-04a0-41e2-99de-26964c329893)

7. После этого подберем лучшие гиперпарметры для модели с эмбедингами из трансформера и для модели с эмбедингами из TF-IDF. Вместе с этим логируем с помозщью MLflow датасеты, установленные параметры модели и сетку параметров, по которой будем искать лучшее сочетание, а также получившиеся метрики. В результате подбора получаем модель и лучшие гиперпараметры, которые также логируем:

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/fc04e523-a285-400f-9e62-86ef8eba9040)

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/7f676b61-96e6-48a7-a41a-f98ec7d7f0f2)

8. После этого с помощью MLflow подгружаем лучшие модели и делаем предсказание. Логирем тестовый датасет и получающиеся метрики:

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/b5680cf5-35fa-4bd6-9208-2155af08723f)

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/18a8a5ac-8027-4080-bca4-83770fdafb47)

10. Все результаты можно посмотреть в MLflow:

   ![image](https://github.com/med-alex/yet_another_mlops_project/assets/118723191/ff8144f2-e472-4bb0-98d9-8fe4c5d7d07a)



