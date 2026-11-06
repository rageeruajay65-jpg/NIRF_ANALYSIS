
const API_URL = "http://127.0.0.1:8000/api/ask";

function useQuestion(question) {
    document.getElementById("questionInput").value = question;
    askQuestion();
}

async function askQuestion() {

    const input = document.getElementById("questionInput");
    const button = document.getElementById("askButton");

    const question = input.value.trim();

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    const loading = document.getElementById("loading");
    const error = document.getElementById("errorMessage");
    const results = document.getElementById("results");

    loading.hidden = false;
    error.textContent = "";
    results.hidden = true;
    button.disabled = true;

    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to analyze question.");
        }

        displayResults(data);

    } catch (err) {

        error.textContent = err.message;

    } finally {

        loading.hidden = true;
        button.disabled = false;

    }
}


// DISPLAY ANSWER, CHART, TABLE AND STATISTICS

function displayResults(data) {

    document.getElementById("results").hidden = false;

    document.getElementById("answerText").textContent =
        data.answer || "No answer available.";

    displayChart(data.chart);

    displayTable(data.table);

    displayStatistics(data.statistics);
}


// DYNAMIC PLOTLY CHART

function displayChart(chartData) {

    const card = document.getElementById("chartCard");

    if (!chartData) {
        card.hidden = true;
        return;
    }

    card.hidden = false;

    document.getElementById("chartHeading").textContent =
        chartData.title || "Data Visualization";

    let trace;

    if (chartData.type === "histogram") {

        trace = {
            x: chartData.x,
            type: "histogram"
        };

    } else if (chartData.type === "scatter") {

        trace = {
            x: chartData.x,
            y: chartData.y,
            mode: "markers",
            type: "scatter"
        };

    } else {

        trace = {
            x: chartData.x,
            y: chartData.y,
            type: chartData.type === "line" ? "scatter" : "bar",
            mode: chartData.type === "line" ? "lines+markers" : undefined
        };

    }

    const layout = {

        title: chartData.title,

        xaxis: {
            title: chartData.x_label
        },

        yaxis: {
            title: chartData.y_label
        },

        autosize: true,

        margin: {
            t: 50,
            l: 60,
            r: 30,
            b: 100
        }

    };

    Plotly.newPlot(
        "chart",
        [trace],
        layout,
        {
            responsive: true,
            displaylogo: false
        }
    );
}


// DYNAMIC TABLE

function displayTable(rows) {

    const card = document.getElementById("tableCard");
    const table = document.getElementById("resultTable");

    table.innerHTML = "";

    if (!rows || rows.length === 0) {
        card.hidden = true;
        return;
    }

    card.hidden = false;

    const columns = Object.keys(rows[0]);

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    columns.forEach(column => {

        const th = document.createElement("th");

        th.textContent = column.replaceAll("_", " ").toUpperCase();

        headerRow.appendChild(th);

    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    rows.forEach(row => {

        const tr = document.createElement("tr");

        columns.forEach(column => {

            const td = document.createElement("td");

            td.textContent = row[column] ?? "-";

            tr.appendChild(td);

        });

        tbody.appendChild(tr);

    });

    table.appendChild(tbody);
}


// DYNAMIC STATISTICS

function displayStatistics(statistics) {

    const card = document.getElementById("statisticsCard");
    const container = document.getElementById("statistics");

    container.innerHTML = "";

    if (!statistics) {
        card.hidden = true;
        return;
    }

    card.hidden = false;

    Object.entries(statistics).forEach(([key, value]) => {

        const item = document.createElement("div");

        item.className = "stat-item";

        const label = document.createElement("p");
        label.textContent = key.replaceAll("_", " ").toUpperCase();

        const strong = document.createElement("strong");
        strong.textContent = value ?? "-";

        item.appendChild(label);
        item.appendChild(strong);

        container.appendChild(item);

    });
}


// ENTER KEY TO SUBMIT

document.getElementById("questionInput").addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter" && !event.shiftKey) {

            event.preventDefault();
            askQuestion();

        }

    }
);
