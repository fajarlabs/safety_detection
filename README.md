## Program deteksi perlengkapan APD menggunakan Yolo dan FastAPI
Deteksi APD 1: <br />
<img src="https://github.com/fajarlabs/safety_detection/blob/master/restapi/static/detected_20241118_153208.jpg" /><br />
Deteksi APD 2: <br />
<img src="https://github.com/fajarlabs/safety_detection/blob/master/restapi/static/detected_20241118_153902.jpg" /><br />
<br /><br />
### Contoh penggunaan APIs
```bash

curl --location 'https://smartdetection.ap.ngrok.io/detect' \
--header 'accept: application/json' \
--form 'file=@"/C:/Users/fajar/Downloads/OIP.jpg"'
```
### Contoh Responsenya
```
{
    "detections": [
        {
            "class": "no-vest",
            "confidence": 0.8565045595169067,
            "box": [
                602,
                368,
                918,
                858
            ]
        },
        {
            "class": "helmet",
            "confidence": 0.8163127899169922,
            "box": [
                389,
                62,
                580,
                200
            ]
        },
        {
            "class": "helmet",
            "confidence": 0.8078612089157104,
            "box": [
                1054,
                138,
                1271,
                288
            ]
        },
        {
            "class": "no-vest",
            "confidence": 0.8070653080940247,
            "box": [
                974,
                381,
                1280,
                909
            ]
        },
        {
            "class": "no-vest",
            "confidence": 0.8065420985221863,
            "box": [
                0,
                369,
                202,
                941
            ]
        },
        {
            "class": "person",
            "confidence": 0.7993049621582031,
            "box": [
                949,
                146,
                1280,
                1280
            ]
        },
        {
            "class": "person",
            "confidence": 0.7898056507110596,
            "box": [
                0,
                90,
                229,
                1280
            ]
        },
        {
            "class": "no-vest",
            "confidence": 0.7826997637748718,
            "box": [
                256,
                316,
                585,
                881
            ]
        },
        {
            "class": "person",
            "confidence": 0.7752936482429504,
            "box": [
                221,
                81,
                599,
                1280
            ]
        },
        {
            "class": "person",
            "confidence": 0.7709600925445557,
            "box": [
                562,
                168,
                960,
                1280
            ]
        },
        {
            "class": "helmet",
            "confidence": 0.691726803779602,
            "box": [
                0,
                98,
                96,
                237
            ]
        },
        {
            "class": "no-helmet",
            "confidence": 0.6419630646705627,
            "box": [
                646,
                168,
                792,
                283
            ]
        },
        {
            "class": "no-helmet",
            "confidence": 0.40943223237991333,
            "box": [
                650,
                169,
                782,
                246
            ]
        }
    ],
    "image_url": "/static/detected_20241118_153902.jpg"
}
```

### Hasil pelatihan dari dataset yang tersedia dengan hasil berikut ini

Confusion Matrix : <br />
<img src="https://github.com/fajarlabs/safety_detection/blob/master/runs/detect/val2/confusion_matrix.png" />
<br />
Confusion Matrix Normalized : <br />
<img  src="https://github.com/fajarlabs/safety_detection/blob/master/runs/detect/val2/confusion_matrix_normalized.png" />
<br />
F1 Curve : <br />
<img  src="https://github.com/fajarlabs/safety_detection/blob/master/runs/detect/val2/F1_curve.png" />
<br />
<br />
