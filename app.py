from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import os, tempfile

app = Flask(__name__)
CORS(app)

ALLOWED_FORMATS = ['mp3', 'wav', 'flac', 'ogg', 'm4a']

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return jsonify({'error': '沒有收到檔案'}), 400
    file = request.files['file']
    target_format = request.form.get('format', 'mp3').lower()
    if target_format not in ALLOWED_FORMATS:
        return jsonify({'error': f'不支援的格式：{target_format}'}), 400
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)
        audio = AudioSegment.from_file(input_path)
        output_filename = os.path.splitext(file.filename)[0] + '.' + target_format
        output_path = os.path.join(tmpdir, output_filename)
        audio.export(output_path, format=target_format)
        return send_file(output_path, as_attachment=True, download_name=output_filename)

@app.route('/trim', methods=['POST'])
def trim():
    if 'file' not in request.files:
        return jsonify({'error': '沒有收到檔案'}), 400
    file = request.files['file']
    start_ms = int(request.form.get('start', 0)) * 1000
    end_ms = int(request.form.get('end', 0)) * 1000
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)
        audio = AudioSegment.from_file(input_path)
        if end_ms == 0 or end_ms > len(audio):
            end_ms = len(audio)
        trimmed = audio[start_ms:end_ms]
        ext = os.path.splitext(file.filename)[1][1:] or 'mp3'
        output_filename = os.path.splitext(file.filename)[0] + '_trimmed.' + ext
        output_path = os.path.join(tmpdir, output_filename)
        trimmed.export(output_path, format=ext)
        return send_file(output_path, as_attachment=True, download_name=output_filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
