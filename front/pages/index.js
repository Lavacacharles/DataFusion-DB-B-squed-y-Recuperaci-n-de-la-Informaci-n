
import { useState } from 'react';
import axios from 'axios';
import styles from '@/styles/Home.module.css';

export default function HomePage() {
  const [inputText, setInputText] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [tableData, setTableData] = useState([]);
  const [selectedOption, setSelectedOption] = useState('postgres');
  const [executionTime, setExecutionTime] = useState(null);
  const [showInfoPopup, setShowInfoPopup] = useState(false);

  

  const parseCSV = (csvText) => {
    const rows = csvText.trim().split('\n');
    return rows.map(row => row.split(','));
  };

  const convertToCSV = (data) => {
    return data.map(row => row.join(',')).join('\n');
  };

  const handleSearch =  () => {
   
    if (!inputText.trim().endsWith(';')) {
      setErrorMessage("La consulta debe terminar con un punto y coma (;)");
      return;
    }

    setErrorMessage('');
    console.log("Consulta válida:", inputText);
    const csvResponse = 'name,age,city,occupation,salary,department\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation\nJohn,30,New York,Engineer,70000,R&D\nJane,25,Boston,Doctor,85000,Health\nDoe,22,San Francisco,Artist,50000,Art\nAlice,29,Chicago,Teacher,60000,Education\nBob,34,Seattle,Nurse,55000,Health\nCharlie,28,Austin,Architect,75000,Construction\nDiana,40,Denver,Scientist,95000,Research\nEve,27,Miami,Lawyer,67000,Law\nFrank,26,Orlando,Chef,52000,Hospitality\nGrace,32,Dallas,Pilot,88000,Aviation';
    try {
      const fetchData = async () => {
        const startTime = performance.now();
        const response = await axios.post('http://localhost:5000/api/search', { query: inputText, index: selectedOption}, {
          headers: { 'Content-Type': 'application/json' },
        });
        const csvData = response.data;
        const data = parseCSV(csvData);
        const endTime = performance.now(); // Tiempo de fin
        setExecutionTime((endTime - startTime).toFixed(2));
        console.log("Tiempo de ejecución:", (endTime - startTime).toFixed(2), "ms");
        console.log("Data:", data);
        return data;
      }
      const data = fetchData();
      setTableData(data);

      //const data = parseCSV(csvResponse);
    } catch (error) {
      console.error("Error al recibir el CSV:", error);
      setErrorMessage("Hubo un problema al recibir los datos del backend.");
    }

    
    const data = parseCSV(csvResponse);
    setTableData(data);

  };

  const handleClear = () => {
    setInputText('');
    setErrorMessage('');
    setTableData([]);
    setExecutionTime(null);
  };
  
  const handleInfoClick = () => {
    setShowInfoPopup(true);
  };
  const handleClosePopup = () => {
    setShowInfoPopup(false);
  };

  const handleDownloadCSV = () => {
    const csv = convertToCSV(tableData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'data.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleOptionChange = (option) => {
    setSelectedOption(option);
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <img src="/utec-logo.png" alt="UTEC Logo" className={styles.logo} />
        <h1 className={styles.title}>Information Retrieval</h1>
        <img src="/ai-icon.png" alt="AI Icon" className={styles.icon} />
      </header>

      <main className={styles.main}>
        <div className={styles.buttonContainer}>
          <button
            className={`${styles.optionButton} ${selectedOption === 'postgres' ? styles.active : ''}`}
            onClick={() => handleOptionChange('postgres')}
          >
            Postgres
          </button>
          <button
            className={`${styles.optionButton} ${selectedOption === 'own' ? styles.active : ''}`}
            onClick={() => handleOptionChange('own')}
          >
            Own Implementation
          </button>
        </div>
        <div className={styles.textAreaWrapper}>
          <textarea
            className={styles.textArea}
            placeholder="Enter your query ..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <div className={styles.buttonGroup}>
            <button className={styles.clearButton} onClick={() => setInputText('')}>
              <img src="/trash-icon.png" alt="Clear" className={styles.clearIcon} />
            </button>
            <button className={styles.infoButton} onClick={handleInfoClick}>
              i
            </button>
          </div>
        </div>

        {errorMessage && <p className={styles.errorMessage}>{errorMessage}</p>}

        {showInfoPopup && (
          <div className={styles.popupOverlay} onClick={handleClosePopup}>
            <div className={styles.popupContent} onClick={(e) => e.stopPropagation()}>
              <h2 className={styles.popupTitle}>Información</h2>
              <p>Ingrese una consulta como el siguiente ejemplo:</p>
              <p className={styles.emptyParagraph}></p>
              <p>SELECT title, artista, lyric FROM Audio WHERE lyric @@ 'amor en tiempos de guerra' LIMIT 10;</p>
              <button onClick={handleClosePopup} className={styles.closePopupButton}>Cerrar</button>
            </div>
          </div>
        )}


        <button className={styles.searchButton} onClick={handleSearch}>
          Search
        </button>
        {executionTime && (
          <p className={styles.executionTime}>
            Tiempo de ejecución: {executionTime} ms
          </p>
        )}

        {tableData.length > 0 && (
          <div className={styles.tableContainer}>
            <div className={styles.scrollableTable}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    {tableData[0].map((header, index) => (
                      <th key={index}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableData.slice(1).map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      <button className={styles.downloadButton} onClick={handleDownloadCSV}>
              Descargar CSV
            </button>

      </main>
    </div>
  );
}
