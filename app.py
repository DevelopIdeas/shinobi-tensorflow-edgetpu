import asyncio
import time
import socketio
from object_detection import ObjectDetection 

loop = asyncio.get_event_loop()
sio = socketio.AsyncClient()
start_timer = None

config = {
    'host': 'localhost',
    'port': '8000',
    'key': '4TLbvfdfu1AG6fum5kxfEQpSb',
    'plug': 'TensorflowEdgeTPU',
    'mode': 'client',
    'type': 'detector',
    'notice': None,
    'connectionType': 'websocket',
    'label_path': './labelmap.txt',
    'model_path': './edgetpu_model.tflite',
    'confidence': 0.85
}

objectDetection = ObjectDetection(config)

async def cx(x):
    global config
    x.update({
        'pluginKey': config['key'],
        'plug': config['plug']
    })
    await sio.emit('ocv', x)

@sio.event
async def connect():
    global config
    print('connected to server')
    await cx({'f':'init','plug':config['plug'],'notice':config['notice'],'type':config['type'],'connectionType':config['connectionType']})

@sio.on('f')
async def on_message(d):
    if d['f'] == "frame":
        await objectDetection.detect(d, cx)
    else:
        print(d['f'])

async def start_server():
    await sio.connect('ws://localhost:8080', transports=["websocket"])
    await sio.wait()

if __name__ == '__main__':
    loop.run_until_complete(start_server())
