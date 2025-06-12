import json
import os
import glob
import shutil
import webbrowser
from jinja2 import Template
from datetime import datetime
from flask import Flask, request, redirect, Response

app = Flask(__name__)

PARAM_FILE = "parameters.json"
BACKUP_DIR = "backups"
VALID_TYPES = ["str", "int", "float", "bool"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Parametry</title>
    <style>
        table { border-collapse: collapse; width: 95%; margin: 20px auto; }
        th, td { padding: 8px 12px; border: 1px solid #444; text-align: center; }
        th { background-color: #eee; }
        .changed { background-color: #f99; font-weight: bold; }
        form { display: inline; }
        input[type=text], select { width: 100px; }
        .icon-btn { background: none; border: none; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h2 style="text-align:center;">Tabela parametrów</h2>
    <table>
        <thead>
            <tr>
                <th>Parametr</th>
                <th>Typ</th>
                <th>Wartość</th>
                <th>Poprzednia wartość</th>
                <th>Opis</th>
                <th>Akcje</th>
            </tr>
        </thead>
        <tbody>
            {{ROWS}}
        </tbody>
    </table>

    <h3 style="text-align:center;">Dodaj nowy parametr</h3>
    <form method="post" action="/save" style="text-align:center; margin-top: 20px;">
        <input type="text" name="key" placeholder="Nazwa parametru" required>
        <select name="type">
            {% for t in types %}
                <option value="{{t}}">{{t}}</option>
            {% endfor %}
        </select>
        <input type="text" name="value" placeholder="Wartość" required>
        <input type="text" name="description" placeholder="Opis" required>
        <button type="submit">Dodaj</button>
    </form>
    {% if error %}
    <p style="color:red; text-align:center;">{{error}}</p>
    {% endif %}
</body>
</html>
"""

def load_parameters():
    if os.path.exists(PARAM_FILE):
        with open(PARAM_FILE, "r") as f:
            return json.load(f)
    return {}

def get_latest_backup():
    files = glob.glob(os.path.join(BACKUP_DIR, "parameters_*.json"))
    if not files:
        return {}
    latest_file = max(files, key=os.path.getmtime)
    with open(latest_file, "r") as f:
        return json.load(f)

def save_backup():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(PARAM_FILE, os.path.join(BACKUP_DIR, f"parameters_{timestamp}.json"))

def escape_html(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def validate_type(value, type_):
    try:
        if type_ == "int":
            int(value)
        elif type_ == "float":
            float(value)
        elif type_ == "bool":
            if value.lower() not in ["true", "false"]:
                raise ValueError("Invalid bool")
        elif type_ == "str":
            str(value)
        else:
            raise ValueError("Unknown type")
        return True
    except:
        return False

def generate_table_rows(current, previous, editing=None):
    rows = ""
    for key, val in current.items():
        val_value = val["value"]
        val_type = val.get("type", "str")
        val_desc = val.get("description", "")
        prev_value = previous.get(key, {}).get("value", "")
        cls = "changed" if val_value != prev_value else ""

        if editing == key:
            rows += f"""
            <tr>
                <form method="post" action="/update">
                    <td>{escape_html(key)}<input type="hidden" name="key" value="{escape_html(key)}"></td>
                    <td>
                        <select name="type">
                            {''.join(f'<option value="{t}" {"selected" if t==val_type else ""}>{t}</option>' for t in VALID_TYPES)}
                        </select>
                    </td>
                    <td><input type="text" name="value" value="{escape_html(val_value)}" required></td>
                    <td>{escape_html(prev_value)}</td>
                    <td><input type="text" name="description" value="{escape_html(val_desc)}" required></td>
                    <td>
                        <button class="icon-btn" type="submit">💾</button>
                        <a href="/" class="icon-btn">❌</a>
                    </td>
                </form>
            </tr>
            """
        else:
            rows += f"""
            <tr>
                <td>{escape_html(key)}</td>
                <td>{escape_html(val_type)}</td>
                <td class="{cls}">{escape_html(val_value)}</td>
                <td>{escape_html(prev_value)}</td>
                <td>{escape_html(val_desc)}</td>
                <td>
                    <form method="get" action="/">
                        <input type="hidden" name="edit" value="{escape_html(key)}">
                        <button class="icon-btn" type="submit">✏️</button>
                    </form>
                    <form method="post" action="/delete" style="display:inline;">
                        <input type="hidden" name="key" value="{escape_html(key)}">
                        <button class="icon-btn" type="submit" onclick="return confirm('Usunąć {key}?')">🗑</button>
                    </form>
                </td>
            </tr>
            """
    return rows

@app.route("/", methods=["GET"])
def index():
    current = load_parameters()
    previous = get_latest_backup()
    editing = request.args.get("edit")
    rows_html = generate_table_rows(current, previous, editing)

    template = Template(HTML_TEMPLATE)
    html = template.render(ROWS=rows_html, types=VALID_TYPES, error=None)
    return Response(html, mimetype="text/html")

@app.route("/save", methods=["POST"])
def save_param():
    key = request.form["key"]
    value = request.form["value"]
    type_ = request.form["type"]
    description = request.form["description"]

    if not validate_type(value, type_):
        return redirect("/?edit=" + key)

    params = load_parameters()
    if key in params:
        return redirect(f"/?edit={key}")

    params[key] = {"value": value, "type": type_, "description": description}
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f, indent=4)
    return redirect("/")

@app.route("/update", methods=["POST"])
def update_param():
    key = request.form["key"]
    value = request.form["value"]
    type_ = request.form["type"]
    description = request.form["description"]

    if not validate_type(value, type_):
        return redirect(f"/?edit={key}")

    params = load_parameters()
    if key in params:
        save_backup()
        params[key] = {"value": value, "type": type_, "description": description}
        with open(PARAM_FILE, "w") as f:
            json.dump(params, f, indent=4)
    return redirect("/")

@app.route("/delete", methods=["POST"])
def delete_param():
    key = request.form["key"]
    params = load_parameters()
    if key in params:
        save_backup()
        del params[key]
        with open(PARAM_FILE, "w") as f:
            json.dump(params, f, indent=4)
    return redirect("/")

if __name__ == "__main__":
    port = 5000
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)
