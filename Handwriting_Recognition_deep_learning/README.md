# Handwriting_Recognition_machine_learning
機器學習手寫辨識數字專案
This project is a example about number_recognizing by tensorflow and django framework.<br>
If you have installed tensorflow and django.<br>
you can use python manage.py runserver to run this project.<br>
The website include two pages:
1.http://127.0.0.1:8000/data_generator/
![](readmepics/data_generator_page.PNG)
2.http://127.0.0.1:8000/index/
you can use this page to predict number,for example if I write a number two.<br>
![](readmepics/predictpage.PNG)
<br>
![](readmepics/example.PNG)
<br>
![](readmepics/result.PNG)
<br>
##Installation Dependencies:

| os| gpu | IDE|  website framework|  
| -- | -- | -- | -- |
| win10 | GTX 1060 | anaconda:3.5.2|django|
<br>
##Installation step:
first step: install CUDA and cuDNN for installing tensorflow1.4
<br>
install tourtial URL:https://www.tensorflow.org/install/install_windows
<br>
two step: pip install -r requirements.txt
<br>
third step : change your cmd path to manage.py folder and run manage.py script<br>

![](readmepics/cmd_runserver.PNG)
<br>python manage.py runserver

![](readmepics/success_local_web.PNG)
