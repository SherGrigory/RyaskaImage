import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from dash import *

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(children='Приложение для анализа фото ряски', style={'text-align': 'center'}),
    html.Hr(),
    dcc.Upload(
        id='Upload_image',
        children='Загрузите фото',
        style={'color': 'blue', 'cursor': 'pointer'}
    ),
    html.Label('Обрежьте картинку сверху'),
    dcc.Slider(min=0, max=2500, step=100, value=0, id='first_slider'),
    html.Label('Обрежьте картинку снизу'),
    dcc.Slider(min=0, max=2500, step=100, value=0, id='second_slider'),
    html.Div(id='output-image-container'),
    html.Div(id='cropped-image', style={'display': 'none'}),
    html.Hr(),
    html.Label('Выберите слой'),
    dcc.Slider(min=0, max=10, step=1, value=1, id='sloy_slayder'),
    html.Div(id='output-image2-container'),
    html.Div(id='cropped-image2', style={'display': 'none'}),
    html.Hr(),
    html.Label('Обрежьте остатки квадрата сверху'),
    dcc.Slider(min=0, max=200, step=10, value=10, id='top_slider'),
    html.Label('Обрежьте остатки квадрата снизу'),
    dcc.Slider(min=0, max=200, step=10, value=10, id='bottom_slider'),
    html.Label('Обрежьте остатки квадрата слева'),
    dcc.Slider(min=0, max=200, step=10, value=10, id='left_slider'),
    html.Label('Обрежьте остатки квадрата справа'),
    dcc.Slider(min=0, max=200, step=10, value=10, id='right_slider'),
    html.Label('Установите нижний уровень, чтобы выделить только ряску'),
    dcc.Slider(min=0, max=255, step=5, value=160, id='level1_slider'),
    html.Label('Установите верхний уровень, чтобы выделить только ряску'),
    dcc.Slider(min=0, max=255, step=5, value=255, id='level3_slider'),
    html.Label('Установите уровень, чтобы обрезать тени (маленькие)'),
    dcc.Slider(min=0, max=800, step=40, value=160, id='level2_slider'),
    html.Div(id='output-image3-container'),
    html.Div(id='duckweed-container')
])


@callback([Output(component_id='output-image-container', component_property='children'),
           Output(component_id='cropped-image', component_property='children')],
          [Input(component_id='Upload_image', component_property='contents'),
           Input(component_id='first_slider', component_property='value'),
           Input(component_id='second_slider', component_property='value')]
          )
def update_output(contents, crop_top, crop_botom):
    if contents is not None:
        # Decode and open the uploaded image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))

        img = img.crop((0, crop_top, img.width, img.height - crop_botom))

        # Encode the resized image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display the resized image
        return [html.Div([
            html.Img(src='data:image/jpeg;base64,{}'.format(img_str), style={'width': '30%', 'height': '30%'})
        ]), img_str]
    else:
        return [None, None]


@callback([Output('output-image2-container', 'children'),
           Output(component_id='cropped-image2', component_property='children')],
          [Input('cropped-image', 'children'),
           Input('sloy_slayder', 'value')])
def process_cropped_image(cropped_image, contnum):
    if cropped_image is not None:
        # Decode base64 image to numpy array
        decoded = base64.b64decode(cropped_image)
        nparr = np.frombuffer(decoded, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Choose a contour (change this according to your needs)
        chosen_contour_index = contnum
        chosen_contour = sorted_contours[chosen_contour_index]

        # Draw contour on the image
        contour_image = img_np.copy()
        cv2.drawContours(contour_image, [chosen_contour], -1, (10, 255, 15), 2)

        # Convert image back to base64 for display
        _, img_encoded = cv2.imencode('.png', contour_image)
        img_base64 = base64.b64encode(img_encoded).decode()

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(chosen_contour)

        # Crop the image based on the bounding box
        countered_image = img_np[y:y + h, x:x + w]

        contour_pil_image = Image.fromarray(countered_image)
        buffered = BytesIO()
        contour_pil_image.save(buffered, format="JPEG")
        img_str2 = base64.b64encode(buffered.getvalue()).decode()

        return [
            html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '30%', 'height': '30%'}),
            img_str2
        ]
    else:
        return [None, None]


@callback([Output('output-image3-container', 'children'),
           Output('duckweed-container', 'children')],
          [Input('cropped-image2', 'children'),
           Input(component_id='top_slider', component_property='value'),
           Input(component_id='bottom_slider', component_property='value'),
           Input(component_id='left_slider', component_property='value'),
           Input(component_id='right_slider', component_property='value'),
           Input(component_id='level1_slider', component_property='value'),
           Input(component_id='level3_slider', component_property='value'),
           Input(component_id='level2_slider', component_property='value')])
def process_cropped_image2(cropped_image2, top, bottom, left, right, level, up_level, size):
    if cropped_image2 is not None:
        # Decode base64 image to numpy array
        decoded = base64.b64decode(cropped_image2)
        nparr = np.frombuffer(decoded, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cropped_image3 = img_np[top:img_np.shape[0] - bottom, left:img_np.shape[1] - right]

        gray2 = cv2.cvtColor(cropped_image3, cv2.COLOR_BGR2GRAY)
        _, thresh2 = cv2.threshold(gray2, level, up_level, cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours by area in descending order
        sorted_contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        chosen_contour2 = sorted_contours2[1:]
        min_contour_area = size  # Adjust this threshold as needed
        # Filter contours based on area
        filtered_contours = [contour for contour in chosen_contour2 if cv2.contourArea(contour) >= min_contour_area]
        contour_image2 = cropped_image3.copy()
        cv2.drawContours(contour_image2, filtered_contours, -1, (10, 255, 15), 2)

        duckweed_area = sum(cv2.contourArea(contour) for contour in filtered_contours)
        pixel_area_cm2 = (5 * 5) / (cropped_image3.shape[1] * cropped_image3.shape[0])
        duckweed_area_cm2 = duckweed_area * pixel_area_cm2
        duckweed_area_text = ("Размер ряски: " + str(duckweed_area_cm2) + " см2")

        # Convert image back to base64 for display
        _, img_encoded = cv2.imencode('.png', contour_image2)
        img_base64 = base64.b64encode(img_encoded).decode()

        return [
            html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '30%', 'height': '30%'}),
            duckweed_area_text
        ]
    else:
        return [None, None]


if __name__ == '__main__':
    app.run(debug=True)
