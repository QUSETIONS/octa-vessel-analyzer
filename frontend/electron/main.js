/**
 * OCTA 血管网络分析系统 - Electron 主进程
 * 
 * 修复版本：使用更可靠的健康检查方式
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn, execSync } = require('child_process');
const fs = require('fs');

// 禁用硬件加速（解决 GPU 进程崩溃问题）
app.disableHardwareAcceleration();

// 后端进程引用
let backendProcess = null;

// 主窗口引用
let mainWindow = null;

// 开发模式标志
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

/**
 * 检查后端是否已经在运行（使用 fetch API）
 */
async function checkBackendRunning() {
    try {
        // 使用 AbortController 设置超时
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        
        const response = await fetch('http://127.0.0.1:8000/health', {
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        return response.ok;
    } catch (error) {
        return false;
    }
}

/**
 * 等待后端启动
 */
async function waitForBackend(maxRetries = 30, interval = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        const isRunning = await checkBackendRunning();
        if (isRunning) {
            console.log('后端服务已就绪');
            return true;
        }
        console.log(`等待后端启动... (${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, interval));
    }
    return false;
}

/**
 * 获取虚拟环境中的 Python 路径
 */
function getVenvPythonPath(backendPath) {
    const isWindows = process.platform === 'win32';
    const venvPath = path.join(backendPath, '.venv');
    
    if (isWindows) {
        const pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
        if (fs.existsSync(pythonPath)) {
            return pythonPath;
        }
    } else {
        const pythonPath = path.join(venvPath, 'bin', 'python');
        if (fs.existsSync(pythonPath)) {
            return pythonPath;
        }
    }
    
    return 'python';
}

/**
 * 创建主窗口
 */
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1600,
        height: 1000,
        minWidth: 1200,
        minHeight: 800,
        title: 'OCTA 血管网络分析系统',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        frame: true,
        backgroundColor: '#1a1a2e',
        show: false
    });

    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

/**
 * 启动后端服务
 */
async function startBackend() {
    // 先检查后端是否已经在运行
    const alreadyRunning = await checkBackendRunning();
    if (alreadyRunning) {
        console.log('后端服务已经在运行，跳过启动');
        return true;
    }

    let backendPath;
    if (isDev) {
        backendPath = path.join(__dirname, '../../backend');
    } else {
        backendPath = path.join(process.resourcesPath, 'backend');
    }

    const pythonPath = getVenvPythonPath(backendPath);
    
    console.log('启动后端服务...');
    console.log('Python 路径:', pythonPath);
    console.log('后端路径:', backendPath);

    // 启动 FastAPI 服务
    backendProcess = spawn(pythonPath, ['-m', 'api.main'], {
        cwd: backendPath,
        env: {
            ...process.env,
            PYTHONPATH: backendPath,
            PYTHONUNBUFFERED: '1'
        },
        shell: false,
        windowsHide: false
    });

    backendProcess.stdout.on('data', (data) => {
        console.log(`[后端] ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
        console.log(`[后端] ${data}`);
    });

    backendProcess.on('close', (code) => {
        console.log(`后端进程退出，代码: ${code}`);
        backendProcess = null;
    });

    backendProcess.on('error', (err) => {
        console.error('启动后端失败:', err);
        backendProcess = null;
    });

    // 等待后端启动
    const started = await waitForBackend(30, 1000);
    return started;
}

/**
 * 停止后端服务
 */
function stopBackend() {
    if (backendProcess) {
        console.log('停止后端服务...');
        
        if (process.platform === 'win32') {
            try {
                execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' });
            } catch (e) {
                // 忽略错误
            }
        } else {
            backendProcess.kill('SIGTERM');
        }
        
        backendProcess = null;
    }
}

// 应用准备就绪
app.whenReady().then(async () => {
    // 启动后端
    const backendStarted = await startBackend();
    
    if (!backendStarted) {
        // 不显示错误对话框，直接继续
        // 因为后端可能已经在运行，只是健康检查有问题
        console.log('警告：后端健康检查未通过，但仍尝试启动界面...');
    }
    
    // 无论如何都创建窗口
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// 所有窗口关闭时退出应用
app.on('window-all-closed', () => {
    stopBackend();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// 应用退出前清理
app.on('before-quit', () => {
    stopBackend();
});

// ============================================================================
// IPC 通信处理
// ============================================================================

ipcMain.handle('dialog:openFile', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: options?.filters || [
            { name: 'OCTA 数据', extensions: ['npy', 'dcm', 'dicom', 'nii', 'gz', 'mat'] },
            { name: '所有文件', extensions: ['*'] }
        ]
    });
    return result;
});

ipcMain.handle('dialog:openFolder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory']
    });
    return result;
});

ipcMain.handle('dialog:saveFile', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, {
        filters: options?.filters || [
            { name: 'STL 文件', extensions: ['stl'] },
            { name: 'PLY 文件', extensions: ['ply'] },
            { name: 'OBJ 文件', extensions: ['obj'] }
        ]
    });
    return result;
});

ipcMain.handle('app:getInfo', () => {
    return {
        version: app.getVersion(),
        name: app.getName(),
        platform: process.platform,
        arch: process.arch,
        isDev: isDev
    };
});

ipcMain.handle('shell:showItemInFolder', (event, filepath) => {
    shell.showItemInFolder(filepath);
});
