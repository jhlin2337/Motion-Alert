import datetime, base64
import json
from flask import Flask, render_template, request, Response
import subprocess
import sys

sys.path.insert(0, 'commands')

import time
import datetime as dt

import io
import CVEnumerations

app = Flask(__name__)

## USEFUL VALUES ##

OUTPUT_IMG_SCALE = 0.5


###################


@app.before_first_request
def do_something_only_once():
    global camera, log
    log = io.open('log.txt', 'wb');
    time.sleep(0.1)
    camera.set_image_scale(OUTPUT_IMG_SCALE)
    camera.switch_cv_operation(CVEnumerations.FACE_DETECTION)
    camera.start_cv_operation()


@app.route("/")
def main():
    global camera
    # Create a template data dictionary to send any data to the template
    templateData = {
        'title': 'Chiem Cam',
        'get_current_cv_operation': camera.get_current_cv_operation(),
        'is_notify_on_motion': camera.get_notify_on_motion,
        'RAW_IMAGE': CVEnumerations.RAW_IMAGE,
        'FACE_DETECTION': CVEnumerations.FACE_DETECTION,
        'MOTION_DETECTION': CVEnumerations.MOTION_DETECTION,
        'CANNY_EDGE_DETECTION': CVEnumerations.CANNY_EDGE_DETECTION,
        'CORNER_DETECTION': CVEnumerations.CORNER_DETECTION,
        'KEYPOINT_DETECTION': CVEnumerations.KEYPOINT_DETECTION
    }

    # Pass the template data into the template picam.html and return it to the user
    return render_template('index.html', **templateData)


@app.route("/cmd/<command>")
def test(command=None):
    print "Received invalid command: " + command
    return "Received invalid command: " + command


## DOESN"T WORK RIGHT NOW
## This is to implement a multipart image stream (more efficient because doesn't require constant
## re-requests of frames. It's just a single stream.
## Generator function for the camera feed
def gen():
    frame = base64.b64encode(camera.sample_image());
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + "data:image/jpeg;base64," + frame + b'\r\n')


@app.route("/cmd/video_feed")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


##########################


@app.route("/cmd/req_picture")
def take_picture():
    img = camera.sample_image_from_operation()

    log.write('Took picture: ' + dt.datetime.now().strftime('%Y-%m-%d at %I.%M.%S %p') + '\n');
    log.flush()

    picture_obj = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d at %I.%M.%S %p"),
        'encoded_picture': base64.b64encode(img),
        'cv_operation': camera.get_current_cv_operation(),
        'notify_motion': camera.get_notify_on_motion()
    }

    # Send notification of motion detection to server
    if camera.get_notify_on_motion():
        picture_obj['motion_detected'] = camera.has_motion_detect()

    return json.dumps(picture_obj)


@app.route("/cmd/clicked_picture")
def click_picture():
    x_pos = request.args.get('x')
    y_pos = request.args.get('y')
    client_img_width = request.args.get('width')
    client_img_height = request.args.get('height')

    return "x: " + str(x_pos) + " - y: " + str(
        y_pos) + " - width: " + client_img_width + " - height: " + client_img_height;

@app.route("/cmd/set_notify_motion")
def set_notify_motion():
    state = request.args.get('bool')
    camera.set_notify_on_motion(state)
    return str(camera.get_notify_on_motion())


@app.route("/cmd/cv_raw_img")
def raw_img():
    camera.switch_cv_operation(CVEnumerations.RAW_IMAGE)
    return "1"


@app.route("/cmd/cv_face_detect")
def face_detect():
    camera.switch_cv_operation(CVEnumerations.FACE_DETECTION)
    return "1"


@app.route("/cmd/cv_motion_detect")
def motion_detect():
    camera.switch_cv_operation(CVEnumerations.MOTION_DETECTION)
    return "1"


@app.route("/cmd/cv_canny_edge_detect")
def canny_edge_detect():
    camera.switch_cv_operation(CVEnumerations.CANNY_EDGE_DETECTION)
    return "1"


@app.route("/cmd/cv_corner_detect")
def corner_detect():
    camera.switch_cv_operation(CVEnumerations.CORNER_DETECTION)
    return "1"
    
@app.route("/cmd/cv_keypoint_detect")
def keypoint_detect():
    camera.switch_cv_operation(CVEnumerations.KEYPOINT_DETECTION)
    return "1"

if __name__ == "__main__":
    # # allow the camera to warmup
    import cvcam as camera

    app.run(debug=True, host='0.0.0.0')

import atexit


def exit_handler():
    camera.stop()
    print "My application is ending!"


atexit.register(exit_handler)
