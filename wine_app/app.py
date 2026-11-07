import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).resolve().parent
ACCENT = "#852D4D"
SECTIONS = ("Оценить вино", "Модельные метрики", "Пакетный прогноз", "Что выучила модель")
FEATURE_LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}
FIELD_DIGITS = {"density": 6, "chlorides": 3, "volatile_acidity": 3, "alcohol": 4}
WINE_TYPES = ("Белое", "Красное")


def revision(path):
    stats = path.stat()
    return stats.st_mtime_ns, stats.st_size


def read_artifact(path, reader, *args):
    try:
        return reader(path, revision(path), *args)
    except Exception:
        st.error(
            f"Проверьте файл `{path.name}` и повторите экспорт модели и данных "
            "из последней ячейки ноутбука ДЗ 3."
        )
        st.stop()


@st.cache_resource
def load_model(path, file_revision):
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["standardscaler", "logisticregression"]:
        raise ValueError("Нужен обученный пайплайн из ноутбука ДЗ 3.")
    if set(model.feature_names_in_) != set(FEATURE_LABELS) or list(model.classes_) != [0, 1]:
        raise ValueError("Схема признаков или классов модели не совпадает с экспортом ДЗ 3.")
    return model


@st.cache_data
def load_threshold(path, file_revision):
    threshold = float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Порог должен быть от 0 до 1.")
    return threshold


def validate_features(frame, features):
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Не хватает столбцов {', '.join(missing)}.")
    extra = [column for column in frame.columns if column not in features]
    if extra:
        raise ValueError(f"Лишние столбцы {', '.join(extra)}.")
    if frame.empty:
        raise ValueError("В файле нет строк с данными.")
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("В строках CSV должно быть столько же значений, сколько столбцов в заголовке.")

    for feature in features:
        values = frame[feature]
        if values.isna().any():
            raise ValueError(f"В столбце `{feature}` есть пропуски.")
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"В столбце `{feature}` есть нечисловые значения.")
        if not np.isfinite(numeric).all():
            raise ValueError(f"В столбце `{feature}` есть бесконечные значения.")
        frame[feature] = numeric
    if not frame["is_red"].isin([0, 1]).all():
        raise ValueError("В столбце `is_red` допустимы только 0 (белое) и 1 (красное).")
    return frame


@st.cache_data
def load_test(path, file_revision, features):
    frame = pd.read_csv(path)
    target = frame.pop("target")
    if target.isna().any() or set(target) != {0, 1}:
        raise ValueError("В тесте должны быть оба класса с метками 0 и 1.")
    return validate_features(frame, features).loc[:, list(features)], target


def probabilities(model, features):
    return model.predict_proba(features)[:, 1]


@st.cache_data
def test_probabilities(_model, features, model_revision):
    return probabilities(_model, features)


@st.cache_data
def curve_points(target, scores):
    fpr, tpr, _ = roc_curve(target, scores)
    precision, recall, _ = precision_recall_curve(target, scores)
    return (fpr, tpr), (recall, precision)


def recommendation(selected):
    return "В подборку" if selected else "Пропустить"


def percent(value):
    return f"{value:.1%}".replace(".", ",")


def shown_threshold(value):
    return f"{value:.4f}".replace(".", ",")


def plot_curve(points, current, x_label, y_label):
    figure = Figure(figsize=(5.2, 3.3), layout="constrained")
    axis = figure.subplots()
    axis.plot(*points, color=ACCENT, linewidth=2)
    axis.scatter(*current, color=ACCENT, s=65, edgecolor="white", zorder=3, label="Текущий порог")
    axis.set(xlabel=x_label, ylabel=y_label, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axis.grid(alpha=0.16)
    axis.legend(loc="best", frameon=False)
    st.pyplot(figure)


def show_metrics(target, scores, threshold):
    st.title("Модельные метрики")
    count = f"{len(target):,}".replace(",", " ")
    st.caption(f"Тестовая выборка · {count} вин")
    selected = scores >= threshold
    tp, fn, fp, tn = confusion_matrix(target, selected, labels=[1, 0]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    fpr = fp / (fp + tn)

    precision_col, recall_col, f1_col = st.columns(3)
    precision_col.metric("Precision", percent(precision), help="Доля хороших среди отобранных вин.")
    recall_col.metric("Recall", percent(recall), help="Доля отобранных среди всех хороших вин.")
    recall_col.caption("Ориентир 70%")
    f1_col.metric("F1", percent(f1), help="Гармоническое среднее precision и recall.")

    st.subheader("Матрица ошибок")
    matrix = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=pd.Index(["Хорошее", "Обычное"], name="Фактический класс"),
        columns=["В подборке", "Пропущено"],
    )
    st.table(matrix)

    roc, pr = curve_points(target, scores)
    roc_col, pr_col = st.columns(2)
    with roc_col:
        st.subheader("ROC")
        plot_curve(roc, (fpr, recall), "Доля ошибочно отобранных обычных вин (FPR)", "Recall")
    with pr_col:
        st.subheader("Precision–Recall")
        plot_curve(pr, (recall, precision), "Recall", "Precision")
    st.caption(
        "При меньшем пороге в подборку попадает больше вин. Мы пропускаем меньше хороших, "
        "но можем отобрать больше обычных. При большем пороге отбор строже."
    )


def show_wine_form(model, test, features, threshold):
    st.title("Оценить вино")
    st.caption("Введите результаты лабораторного анализа.")
    form_col, result_col = st.columns([2, 1])
    last = st.session_state.get("wine_result")
    defaults = last["values"] if last else test.median()
    values = {}

    with form_col, st.form("wine_form"):
        wine_type = st.radio("Тип вина", WINE_TYPES, index=int(defaults["is_red"]) if last else 0, horizontal=True)
        values["is_red"] = WINE_TYPES.index(wine_type)
        columns = st.columns(2)
        for index, feature in enumerate(name for name in features if name != "is_red"):
            digits = FIELD_DIGITS.get(feature, 2)
            values[feature] = columns[index % 2].number_input(
                FEATURE_LABELS[feature],
                min_value=float(test[feature].min()),
                max_value=float(test[feature].max()),
                value=float(defaults[feature]),
                step=10.0 ** -digits,
                format=f"%.{digits}f",
                key=f"wine_{feature}",
            )
        submitted = st.form_submit_button("Оценить вино", type="primary")

    if submitted:
        row = pd.DataFrame([values], columns=list(features))
        last = {"values": values, "probability": float(probabilities(model, row)[0])}
        st.session_state["wine_result"] = last

    with result_col, st.container(border=True):
        st.subheader("Результат")
        if last is None:
            st.write("Заполните форму и нажмите «Оценить вино».")
            return
        score = last["probability"]
        st.metric("Вероятность хорошего вина", percent(score))
        st.write(recommendation(score >= threshold))
        st.caption(f"Порог отбора {shown_threshold(threshold)}")


def read_batch(file_bytes, features):
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV-файл пуст.") from error
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError("Не удалось прочитать файл. Сохраните CSV с запятой в кодировке UTF-8.") from error
    return validate_features(frame, features)


def handle_upload(model, features):
    st.session_state.pop("batch_result", None)
    st.session_state.pop("batch_error", None)
    uploaded = st.session_state.get("batch_upload")
    if uploaded is None:
        return
    try:
        frame = read_batch(uploaded.getvalue(), features)
        scores = probabilities(model, frame.loc[:, list(features)])
    except ValueError as error:
        st.session_state["batch_error"] = str(error)
        return
    st.session_state["batch_result"] = (uploaded.name, frame, scores)


def show_batch(model, features, threshold):
    st.title("Пакетный прогноз")
    st.caption("Загрузите CSV с 12 признаками вин. Для всей партии применяется общий порог отбора.")
    st.file_uploader("Выбрать CSV", type="csv", key="batch_upload", on_change=handle_upload, args=(model, features))
    with st.expander("Столбцы CSV"):
        st.code(",".join(features), language=None, wrap_lines=True)
        st.caption("Укажите 0 для белого вина и 1 для красного в столбце is_red.")

    error = st.session_state.get("batch_error")
    if error:
        st.error(error)
    result = st.session_state.get("batch_result")
    if result is None:
        return
    name, frame, scores = result
    output = frame.assign(
        probability_good=scores,
        recommendation=[recommendation(selected) for selected in scores >= threshold],
    )
    st.caption(f"{name} · {len(output)} вин")
    st.dataframe(
        output,
        hide_index=True,
        column_config={
            "probability_good": st.column_config.NumberColumn("Вероятность хорошего вина", format="percent"),
            "recommendation": "Решение",
        },
    )
    st.download_button(
        "Скачать CSV",
        output.to_csv(index=False).encode("utf-8-sig"),
        file_name="wine_predictions.csv",
        mime="text/csv",
        on_click="ignore",
    )


def show_coefficients(model, features):
    st.title("Что выучила модель")
    st.caption("Коэффициенты после масштабирования признаков.")
    weights = pd.Series(model.named_steps["logisticregression"].coef_[0], index=list(features)).sort_values()
    labels = [FEATURE_LABELS[feature] for feature in weights.index]
    figure = Figure(figsize=(7.5, 5), layout="constrained")
    axis = figure.subplots()
    axis.barh(labels, weights, color=[ACCENT if value > 0 else "#52647A" for value in weights])
    axis.axvline(0, color="#292027", linewidth=1)
    axis.set_xlabel("Коэффициент логистической регрессии")
    st.pyplot(figure)
    positive = ", ".join(FEATURE_LABELS[key].lower() for key in weights[weights > 0].nlargest(2).index)
    negative = ", ".join(FEATURE_LABELS[key].lower() for key in weights[weights < 0].nsmallest(2).index)
    st.write(f"Сильнее всего повышают оценку модели — {positive}. Сильнее всего снижают — {negative}.")
    st.caption("Знак коэффициента показывает направление изменения оценки при прочих равных.")


def main():
    st.set_page_config(page_title="Винная подборка", layout="wide")
    model_path = APP_DIR / "model.joblib"
    model = read_artifact(model_path, load_model)
    model_revision = revision(model_path)
    features = tuple(model.feature_names_in_)
    test, target = read_artifact(APP_DIR / "test.csv", load_test, features)
    default_threshold = read_artifact(APP_DIR / "config.json", load_threshold)

    st.sidebar.title("Винная подборка")
    threshold = st.sidebar.slider("Порог отбора", 0.0, 1.0, default_threshold, step=0.0001, format="%.4f", key="threshold")
    st.sidebar.caption(f"Один порог для всех прогнозов · {shown_threshold(threshold)}")
    section = st.sidebar.radio("Раздел", SECTIONS, index=1, key="section")

    if section == SECTIONS[0]:
        show_wine_form(model, test, features, threshold)
    elif section == SECTIONS[1]:
        scores = test_probabilities(model, test, model_revision)
        show_metrics(target, scores, threshold)
    elif section == SECTIONS[2]:
        show_batch(model, features, threshold)
    else:
        show_coefficients(model, features)


if __name__ == "__main__":
    main()
