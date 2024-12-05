import "@/styles/globals.css";
import axios from 'axios';
import { useState, useEffect } from 'react';
import styles from '@/styles/app.module.css';

export default function App({ Component, pageProps }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState(''); 
  const [toSignIn, setToSignIn] = useState(true);
  const [toSignUp, setToSignUp] = useState('');

  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const [inputText, setInputText] = useState('');
  const [limit, setLimit] = useState('');
  
  const [folderContent, setFolderContent] = useState([]);
  const [showTables, setShowTables] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [expandedTables, setExpandedTables] = useState({}); 

  const [selectedOption, setSelectedOption] = useState('search');
  const [selectedType, setSelectedType] = useState("text");
  const [selectedIndex, setSelectedIndex] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [tableData, setTableData] = useState([]);
  const [executionTime, setExecutionTime] = useState(null);
  const [listDataset, setListDataset] = useState([{}]);
  const [messageSuccess, setMessageSuccess] = useState('');
  const [file, setFile] = useState(null);
  const [indexes, setIndexes] = useState([]);

  /**
   * 
   * 
   * 
   */

  const getIndexKey = () => {
    if (selectedIndex === 'postgres') {
      return 'gin';
    } else if (selectedIndex === 'own') {
      return 'own';
    }
    return null;
  };

  const indexKey = getIndexKey();
  const datasets = indexKey ? indexes[indexKey] : [];
  const SelectOptionsSearch = (indexes) => {
    useEffect(() => {
      if (datasets.length > 0) {
        //setSelectedDataset(datasets[0].name);
      } else {
        setSelectedDataset('');
      }
    }, [indexKey, datasets]);
  
    return (
      <div className={styles.selectContainer}>
        {/* Select de Tipo */}
        <select
          className={styles.selectFormat}
          value={selectedType}
          onChange={(e) => {
            setSelectedType(e.target.value);
            setSelectedIndex(''); // Reinicia el índice cuando cambia el tipo
            setSelectedDataset(''); // Reinicia el dataset cuando cambia el tipo
          }}
        >
          <option value="text">TEXT</option>
          <option value="image">IMAGE</option>
        </select>
  
        {/* Select de Índice */}
        <select
          className={styles.selectFormat}
          value={selectedIndex}
          onChange={(e) => {
            setSelectedIndex(e.target.value);
            setSelectedDataset(''); // Reinicia el dataset cuando cambia el índice
          }}
        >
          <option value="" disabled selected hidden>
            INDEX
          </option>
          {selectedType === 'text' ? (
            <>
              <option value="own">OWN</option>
              <option value="postgres">POSTGRES</option>
            </>
          ) : (
            <>
              <option value="seq">KNN-SEQ</option>
              <option value="rtree">RTREE</option>
              <option value="hd">HIGH-D</option>
            </>
          )}
        </select>
  
        {/* Select de Dataset */}
        {selectedType === 'text' && indexKey && (
          <select
            className={styles.selectFormat}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            <option value="" disabled selected hidden>
              DATASET
            </option>
            {datasets.map((data, index) => (
              <option key={index} value={data.name}>
                {data.name}
              </option>
            ))}
          </select>
        )}
      </div>
    );
  };

  /**
   * 
   * 
   * 
   * 
   * 
   */
  const handleRegister = async () => {
    setLoading(true);
    setError(null);
    setMessageSuccess(null);
    try {
      const response = await axios.post('http://localhost:5000/register', { username: username, password:password }, {
        headers: { 'Content-Type': 'application/json' },
      });

      console.log("Response",response)

      if (response.status === 200) {
        setIsAuthenticated(false);
        setToSignUp(false);
        setToSignIn(true);
        setUser('');
        setPassword('')
      }
      
    } catch (error){
      setError(
        error.response?.data?.message || "Ocurrió un error inesperado. Intenta nuevamente."
      ); // Maneja el error y muestra el mensaje
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async() => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/login', { username: username, password:password }, {
        headers: { 'Content-Type': 'application/json' },
      });
      const folder = response.data.folder; 
      const indxs = response.data.indexes;
      console.log("Archivos recibidos:", folder);
      console.log("Archivos recibidos:", indxs);
      setIndexes(indxs);
      setFolderContent(folder);
      setIsAuthenticated(true);   
    } catch (error){
      setError(
        error.response?.data?.message || "Ocurrió un error inesperado. Intenta nuevamente."
      )// Maneja el error y muestra el mensaje
    } finally {
      setLoading(false);
    }
  };

  const handleToSignIn = () =>{
    console.log("Sign In ACTIVATE")
    setToSignIn(true);
    setToSignUp(false);
  };

  const handleToSignUp = () =>{
    console.log("Sign UP ACTIVATE")
    setToSignIn(false);
    setToSignUp(true);
  };

  const handleLogout = () =>{
    setErrorMessage('')
    setFolderContent('')
    setTableData('')
    setSelectedOption('search')
    setSelectedType('text')
    setIsAuthenticated(false);
    setPassword('');
    setUser('')
    setExecutionTime('')

  };

  const handleQuery = (queryFile)  => {
    console.log(queryFile.content)
  }

  const handleClear = () =>{
    setTableData([]);
    setInputText('');
    setExecutionTime('')
  };

  const handleInfoClick = () =>{}


  const handleOptionChange = (option) => {
    setSelectedOption(option);
  };

  const handleRun = async () => {
    let dataToSend = {};
  
    if (selectedOption === "search") {
      dataToSend = {
        username:username,
        type: selectedOption,
        inputText:inputText,
        format: selectedType,
        dataset: selectedDataset,
        indexes: indexes,
        index: selectedIndex,
        limit:limit,
      };
    } else {
      dataToSend = {
        username:username,
        type: selectedOption,
        inputText:inputText,
        limit:limit,
      };
    }
  
    console.log("Consulta válida:", inputText);
    console.log("Data a enviar", dataToSend);
  
    setLoading(true);
    setError(null);
  
    try {
      const startTime = performance.now();
      const response = await axios.post("http://localhost:5000/run", dataToSend, {
        headers: { "Content-Type": "application/json" },
      });
      const endTime = performance.now();
      setExecutionTime((endTime - startTime).toFixed(2));
  
      if (response.data.result) {
        const csvData = response.data.result;
        const indxs = response.data.indexes;
        console.log("Data indexes:", indxs)
        console.log("Data mostrar:", csvData)
        setIndexes(indxs);
        setTableData(parseCSV(csvData));
      } else {
        setTableData([]);
      }
  
      if (response.data.folder) {
        setFolderContent(response.data.folder);
      }
  
      console.log("Tiempo de ejecución:", (endTime - startTime).toFixed(2), "ms");
    } catch (error) {
      console.error("Error en la solicitud:", error);
      setError(
        error.response?.data?.message || "Ocurrió un error inesperado. Intenta nuevamente."
      ); 
    } finally {
      setLoading(false);
    }
    {error && (
      <Popup
        title="Error"
        message={error}
        onClose={() => setError(null)}
      />
    )}
  };

  const parseCSV = (csvText) => {
    const rows = csvText.trim().split('\n');
    return rows.map(row => row.split('%'));
  };

  const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;  
    }
  };

  const isImageUrl = (url) => {
    return /\.(jpeg|jpg|gif|png|bmp|svg)$/i.test(url);
  };

  const handleDragStart = (event, fileUrl) => {
    event.dataTransfer.setData('text/plain', fileUrl);
  };
  
  const handleDrop = (event) => {
    event.preventDefault(); 
    const fileUrl = event.dataTransfer.getData('text/plain');
    setInputText(fileUrl);
    event.target.value = fileUrl;
  };

  const sendCsvToBackend = async (csvFile) => {
    const backendUrl = 'http://localhost:5000/upload-csv';  
    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('username', username)
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(backendUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
  
      if (response.status === 200) {
        console.log('CSV enviado exitosamente:', response.data);
      }
      return response

    } catch (error) {
      console.error('Error al enviar el archivo CSV al backend:', error);
      setError(
        error.response?.data?.message || "Ocurrió un error inesperado. Intenta nuevamente."
      )
    } finally {
      setLoading(false);
    }

  };

  const saveFileToDatabase = async (userName, fileName, typeFile, fileUrl) => {
    const backendUrl = 'http://localhost:5000/saveFile';
  
    try {
      const response = await axios.post(backendUrl, {
        name: fileName,
        url: fileUrl,
        username: userName,
        type: typeFile
      }, {
        headers: { "Content-Type": "application/json" },
      });
  
      return response;
    } catch (error) {
      console.error('Error al guardar en la base de datos:', error);
      return { ok: false }; // Devuelve un estado de error
    }
  };

  const FolderViewer = ({ folder }) => {
    const toggleMainSection = () => {
      setShowTables((prevShowTables) => {
        if (prevShowTables) {
          setExpandedTables({});
        }
        return !prevShowTables;
      });
    };

    const toggleTableSection = (tableName, section) => {
      setExpandedTables((prevState) => {
        const isExpanded = prevState[tableName]?.[section];
    
        // Si el padre se oculta, también colapsa los hijos
        if (section === 'expanded' && isExpanded) {
          return {
            ...prevState,
            [tableName]: {
              expanded: !isExpanded, // Alterna el estado del padre
              showColumns: false,   // Colapsa columnas
              showIndexes: false,   // Colapsa índices
            },
          };
        }
    
        // De lo contrario, solo alterna la sección específica
        return {
          ...prevState,
          [tableName]: {
            ...prevState[tableName],
            [section]: !isExpanded,
          },
        };
      });
    }

    return (
      <div className={styles.folderContainer}>
        <h3 className={styles.titleFolder}>
          📁 @{username}
        </h3>
        <div className={styles.showTablesContainer}>
          <button onClick={toggleMainSection} className={styles.showTablesButton} >
            {showTables ? '▾' : '▸'}
            <img src="/images/table-view.png" className={styles.iconTablesButton} />
            Tablas
          </button>
          {showTables && (
            <div className={styles.tableItemContainer}>
              {folder.tables.map((table, index) => (
                <ul key={index}>
                <li className={styles.tableItem}>
                  <button
                    onClick={() => toggleTableSection(table.tableName, 'expanded')}
                    className={styles.showPropertiesButton}
                  >
                    {expandedTables[table.tableName]?.expanded ? '▾' : '▸'}
                    <img src="/images/table.png" className={styles.iconTablesButton} />
                    {table.tableName}
                  </button>
                  
                  {expandedTables[table.tableName]?.expanded && (
                    <div className={styles.tablePropertiesContainer}>
                      {table.columns.length > 0 && (
                        <div>
                          <button
                            onClick={() => toggleTableSection(table.tableName, 'showColumns')}
                            className={styles.showPropertiesButton}
                          >
                            {expandedTables[table.tableName]?.showColumns ? '▾' : '▸'}
                            <img src="/images/columns.png" className={styles.iconTablesButton} />
                            Columnas
                          </button>
                          {expandedTables[table.tableName]?.showColumns && (
                            <div className={styles.tableColumnsContainer}>
                              <ul >
                                {table.columns.map((col, colIndex) => (
                                  <li key={colIndex} className={styles.tableColumns}>
                                    <img src="/images/column.png" className={styles.iconTablesButton} />
                                    {col}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                      {table.hasIndex && (
                        <div>
                          <button
                            onClick={() => toggleTableSection(table.tableName, 'showIndexes')}
                            className={styles.showPropertiesButton}
                          >
                            {expandedTables[table.tableName]?.showIndexes ? '▾' : '▸'}
                            <img src="/images/indexes.png" className={styles.iconTablesButton} />
                            Indices
                          </button>
                          {expandedTables[table.tableName]?.showIndexes && (
                            <div className={styles.tableIndexesContainer}>
                              <ul>
                                {table.index.map((idx, idxIndex) => (
                                  <li key={idxIndex} className={styles.tableIndexes}>
                                    <img src="/images/index.png" className={styles.iconTablesButton} />
                                    {idx.name}{Array.isArray(idx.columns)? `[${idx.columns.join(", ")}]`: `[${idx.columns}]`} ({idx.by})
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </li>
              </ul>
              ))}
            </div>
          )}
        </div>
  
        <div>
          <div className={styles.showFilesContainer}>
            <button onClick={() => setShowFiles(!showFiles)} className={styles.showFiles}>
              {showFiles ? '▾':'▸'}
              <img src="/images/files.png" className={styles.iconTablesButton} />
              Files
            </button>
            {showFiles && (
              <div className={styles.itemsFileContainer}>
                <ul >
                  {folder.files.map((file, index) => (
                    <li key={index} className={styles.fileItem} draggable="true" onDragStart={(e) => handleDragStart(e, file.url)}>
                      <img src={
                        (file.type === 'image/jpeg' || file.type === 'image/png' ||file.type === 'image/jpg') 
                          ? '/images/image-icon.png'
                          : file.type === 'text/csv'
                          ? '/images/csv-icon.png'
                          : '/images/file-icon.png'
                      }className={styles.fileIcon} />
                      <button className={styles.fileName}>{file.filename}</button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const uploadImage = async (event) => {
    const selectedFile = event.target.files[0]; 
    if (!selectedFile) {
      setError('No se seleccionó una imagen. Por favor selecciona una antes de subir.');
      return;
    }

    const fileType = selectedFile.type; 
    const fileName = selectedFile.name; 
    
    if (fileType === 'text/csv' || fileName.endsWith('.csv')) {
      console.log('Archivo CSV detectado:', fileName);
      const response = await sendCsvToBackend(selectedFile);
      if (response.data.folder) {
        setFolderContent(response.data.folder);
      }
      return;
    }

    console.log('Nombre del archivo:', fileName);
    console.log('Tipo del archivo:', fileType);
    setError(null);
    setLoading(true); 
    const formData = new FormData();
    formData.append('file', selectedFile); 
    formData.append('upload_preset', 'parcial_proyecto'); 
  
    const cloudinaryUrl = 'https://api.cloudinary.com/v1_1/dzli6ozmk/image/upload';
  
    try {
      const response = await axios.post(cloudinaryUrl, formData);
      const imageUrl = response.data.secure_url; // Obtiene la URL pública
      console.log('URL pública:', imageUrl);
      const registerFile = await saveFileToDatabase(username,fileName,fileType,imageUrl)
      if (registerFile.data.folder) {
        setFolderContent(registerFile.data.folder);
      }

    } catch (error) {
      console.error('Error en la subida:', error);
      setError(
        error.response?.data?.message || "Ocurrió un error inesperado. Intenta nuevamente."
      )
    } finally {
      setLoading(false);
    }
  };
  
  const Popup = ({ title, message, onClose }) => {
    return (
      <div className={styles.popupOverlay}>
        <div className={styles.popupContainer}>
          <div className={styles.popupHeader}>
            <h2>{title}</h2>
            <button className={styles.popupCloseBtn} onClick={onClose}>
              &times;
            </button>
          </div>
          <div className={styles.popupBody}>
            <p>{message}</p>
          </div>
        </div>
      </div>
    );
  };

  const Loader = () => {
    return (
      <div className={styles.popupOverlay}>
        <div className={styles.loaderContainer}>
          <div className={styles.loader}></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  };
  

  if (!isAuthenticated && toSignIn) {
    return (
      <div className={styles.container}>
        <header className={styles.header}>
          <img src="/images/utec-logo.png" alt="UTEC Logo" className={styles.logo} />
          <h1 className={styles.title}>Information Retrieval</h1>
          <img src="/images/ai-icon.png" alt="AI Icon" className={styles.icon} />
        </header>
        <main>
          <div className={styles.signContainer}>
            <button onClick={handleToSignIn} className={styles.signinButton}>
              Sign In
            </button>
            <button onClick={handleToSignUp} className={styles.signupButton}>
              Sign Up
            </button>

          </div>
          <div className={styles.loginContainer}>
            <div>
              <h2 className={styles.titleLogin}>
                Login
              </h2>
            </div>
            <div className={styles.loginForm}>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUser(e.target.value)}
                className={styles.loginInput}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={styles.loginInput}
              />
              <button onClick={handleLogin} className={styles.loginButton}>
                Login
              </button>
              {loading && <Loader />}
              {error && (
                <Popup
                  title="Error"
                  message={error}
                  onClose={() => setError(null)}
                />
              )}
            </div>
          </div>
        </main>
      </div>
    );
  }

  else if (toSignUp && !toSignIn) {
    return (
      <div className={styles.container}>
        <header className={styles.header}>
          <img src="/images/utec-logo.png" alt="UTEC Logo" className={styles.logo} />
          <h1 className={styles.title}>Information Retrieval</h1>
          <img src="/images/ai-icon.png" alt="AI Icon" className={styles.icon} />
        </header>
        <main>
          <div className={styles.signContainer}>
            <button onClick={handleToSignIn} className={styles.signinButton}>
              Sign In
            </button>
            <button onClick={handleToSignUp} className={styles.signupButton}>
              Sign Up
            </button>

          </div>
          <div className={styles.loginContainer}>
            <div>
              <h2 className={styles.titleLogin}>
                New Account
              </h2>
            </div>
            <div className={styles.loginForm}>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUser(e.target.value)}
                className={styles.loginInput}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={styles.loginInput}
              />
              <button onClick={handleRegister} className={styles.registerButton}>
                Register
              </button>
              {loading && <Loader />}
              {error && (
                <Popup
                  title="Error"
                  message={error}
                  onClose={() => setError(null)}
                />
              )}
            </div>
          </div>
        </main>
      </div>
    );
  }


  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <img src="/images/utec-logo.png" alt="UTEC Logo" className={styles.logo} />
        <h1 className={styles.title}>Information Retrieval</h1>
        <img src="/images/ai-icon.png" alt="AI Icon" className={styles.icon} />
      </header>
      <main >
        <div className={styles.welcomeContainer}>
          <p className={styles.welcomeText}>
            Bienvenido {username}!
          </p>
          <button className={styles.logoutButton} onClick={handleLogout}>
            <img src="/images/logout.png" className={styles.logoutIcon} />
            <p className={styles.logoutText}>Logout</p>
          </button>
        </div>
        <div className={styles.appContainer}>
          <div className={styles.dataContainer}>
            <FolderViewer folder={folderContent} />
            <div className={styles.uploadContainer}>
              <input
                type="file"
                id="fileInput"
                style={{ display: 'none' }}
                onChange={uploadImage}
              />
              <button
                className={styles.uploadButton}
                onClick={() => document.getElementById('fileInput').click()}
              >
                <img src="/images/upload.png" className={styles.runIcon} alt="Upload Icon" />
                <span className={styles.runText}>Upload</span>
              </button>
              {loading && <Loader />}
              {error && (
                <Popup
                  title="Error"
                  message={error}
                  onClose={() => setError(null)}
                />
              )}

            </div>
          </div>
          <div className={styles.workContainer}>
            <div className={styles.scriptsContainer}>
              <div className={styles.choiceImplementationContainer}>
                <div className={styles.choiceWrapper}>
                  <button
                    className={`${styles.optionChoiceButton} ${selectedOption === 'search' ? styles.choiceActive : ''}`}
                    onClick={() => handleOptionChange('search')}
                  >
                    SEARCH
                  </button>
                  <button
                    className={`${styles.optionChoiceButton} ${selectedOption === 'create' ? styles.choiceActive : ''}`}
                    onClick={() => handleOptionChange('create')}
                  >
                    CREATE
                  </button>
                </div>
              </div>
              {selectedOption === 'search' && (
                <div className={styles.searchContainer}>
                <input className={styles.searchQuery} placeholder="SEARCH" type="text" onChange={(e) => setInputText(e.target.value)} onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} />
                <input className={styles.limitQuery} placeholder="LIMIT" type="number" onChange={(e) => setLimit(e.target.value)} />
                <SelectOptionsSearch indexes={indexes}/>
                </div>
              )}
              {selectedOption === 'create' && (
                <div className={styles.wrapperQuery}>
                <textarea
                  className={styles.textArea}
                  placeholder="Enter your query ..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                />
                <div className={styles.buttonGroup}>
                  <button className={styles.clearButton} onClick={handleClear}>
                    <img src="/images/trash-icon.png" alt="Clear" className={styles.clearIcon} />
                  </button>
                </div>
                </div>
              )}
            </div>
            <div className={styles.runContainer}>
              <button className={styles.runButton} onClick={handleRun}>
                <img src="/images/play.png"  className={styles.runIcon} />
                <p className={styles.runText}>Run</p>
              </button>
              {loading && <Loader />}
              {error && (
                <Popup
                  title="Error"
                  message={error}
                  onClose={() => setError(null)}
                />
              )}
            </div>
            <div className={(selectedOption === 'search')?styles.viewTableContainerSearch:styles.viewTableContainerCreate}>
              {tableData.length > 0 && (
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
                          <td key={cellIndex}>
                            {isValidUrl(cell) ? (
                              isImageUrl(cell) ? (
                                <a href={cell} target="_blank" rel="noopener noreferrer">
                                  <img 
                                    src={cell} 
                                    alt={`Imagen ${rowIndex + 1}`} 
                                    style={{ width: 'auto', height: '200px' }} 
                                  />
                                </a>
                              ) : (
                                <a href={cell} target="_blank" rel="noopener noreferrer">
                                  {cell}
                                </a>
                              )
                            ) : (
                              cell
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>)}
            </div>
            <div className={styles.executeContainer}>
              {executionTime && (
                <p className={styles.executionTime}>
                  Tiempo de ejecución: {executionTime} ms
                </p>
              )}
              <p className={styles.credits}>
                  @bmastree 
                </p>
            </div>
          </div>
          
        </div>
      </main>
    </div>
  );
}
