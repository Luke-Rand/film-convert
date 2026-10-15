#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <io.h>
#include <fcntl.h>

typedef uint32_t EdsError;
typedef void* EdsBaseRef;
typedef EdsBaseRef EdsCameraListRef;
typedef EdsBaseRef EdsCameraRef;
typedef EdsBaseRef EdsStreamRef;
typedef EdsBaseRef EdsEvfImageRef;

typedef EdsError (__stdcall *FnEdsInitializeSDK)(void);
typedef EdsError (__stdcall *FnEdsTerminateSDK)(void);
typedef EdsError (__stdcall *FnEdsGetCameraList)(EdsCameraListRef *outCameraList);
typedef EdsError (__stdcall *FnEdsGetChildCount)(EdsBaseRef inRef, uint32_t *outCount);
typedef EdsError (__stdcall *FnEdsGetChildAtIndex)(EdsBaseRef inRef, int32_t inIndex, EdsBaseRef *outRef);
typedef EdsError (__stdcall *FnEdsOpenSession)(EdsCameraRef inCameraRef);
typedef EdsError (__stdcall *FnEdsCloseSession)(EdsCameraRef inCameraRef);
typedef EdsError (__stdcall *FnEdsSetPropertyData)(EdsBaseRef inRef, uint32_t inPropertyID, int32_t inParam, uint32_t inSize, const void *inData);
typedef EdsError (__stdcall *FnEdsCreateMemoryStream)(uint64_t inBufferSize, EdsStreamRef *outStream);
typedef EdsError (__stdcall *FnEdsCreateEvfImageRef)(EdsStreamRef inStreamRef, EdsEvfImageRef *outEvfImageRef);
typedef EdsError (__stdcall *FnEdsDownloadEvfImage)(EdsCameraRef inCameraRef, EdsEvfImageRef inEvfImageRef);
typedef EdsError (__stdcall *FnEdsGetPointer)(EdsStreamRef inStreamRef, void **outPointer);
typedef EdsError (__stdcall *FnEdsGetLength)(EdsStreamRef inStreamRef, uint64_t *outLength);
typedef uint32_t (__stdcall *FnEdsRelease)(EdsBaseRef inRef);
typedef EdsError (__stdcall *FnEdsSendCommand)(EdsCameraRef inCameraRef, uint32_t inCommand, int32_t inParam);

static volatile int g_running = 1;
static EdsCameraRef g_cam = NULL;
static FnEdsSetPropertyData g_pSetProp = NULL;
static FnEdsSendCommand g_pSendCmd = NULL;

DWORD WINAPI StdinThread(LPVOID param) {
    char line[256];
    while (g_running) {
        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }
        if (strncmp(line, "QUIT", 4) == 0) {
            g_running = 0;
            break;
        } else if (strncmp(line, "FOCUS", 5) == 0) {
            int step = 0;
            if (sscanf(line + 5, "%d", &step) == 1 && g_cam && g_pSetProp) {
                uint32_t drive = 0;
                if (step > 0) {
                    drive = (step >= 3) ? 0x0003 : (step >= 2 ? 0x0002 : 0x0001);
                } else if (step < 0) {
                    drive = (step <= -3) ? 0x8003 : (step <= -2 ? 0x8002 : 0x8001);
                }
                if (drive != 0) {
                    g_pSetProp(g_cam, 0x00000503, 0, sizeof(drive), &drive);
                }
            }
        } else if (strncmp(line, "SHUTTER", 7) == 0) {
            if (g_cam && g_pSendCmd) {
                g_pSendCmd(g_cam, 0, 0); // kEdsCameraCommand_TakePicture
            }
        }
    }
    return 0;
}

int main() {
    // Set binary mode for stdout
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    SetDllDirectoryA("C:\\Program Files (x86)\\Canon\\EOS Utility\\EU3");
    HMODULE hMod = LoadLibraryA("C:\\Program Files (x86)\\Canon\\EOS Utility\\EU3\\EDSDK.dll");
    if (!hMod) {
        fprintf(stderr, "[Canon Bridge] Failed to load EDSDK.dll (err=%lu)\n", GetLastError());
        return 1;
    }

    FnEdsInitializeSDK pEdsInitializeSDK = (FnEdsInitializeSDK)GetProcAddress(hMod, "EdsInitializeSDK");
    FnEdsTerminateSDK pEdsTerminateSDK = (FnEdsTerminateSDK)GetProcAddress(hMod, "EdsTerminateSDK");
    FnEdsGetCameraList pEdsGetCameraList = (FnEdsGetCameraList)GetProcAddress(hMod, "EdsGetCameraList");
    FnEdsGetChildCount pEdsGetChildCount = (FnEdsGetChildCount)GetProcAddress(hMod, "EdsGetChildCount");
    FnEdsGetChildAtIndex pEdsGetChildAtIndex = (FnEdsGetChildAtIndex)GetProcAddress(hMod, "EdsGetChildAtIndex");
    FnEdsOpenSession pEdsOpenSession = (FnEdsOpenSession)GetProcAddress(hMod, "EdsOpenSession");
    FnEdsCloseSession pEdsCloseSession = (FnEdsCloseSession)GetProcAddress(hMod, "EdsCloseSession");
    g_pSetProp = (FnEdsSetPropertyData)GetProcAddress(hMod, "EdsSetPropertyData");
    FnEdsCreateMemoryStream pEdsCreateMemoryStream = (FnEdsCreateMemoryStream)GetProcAddress(hMod, "EdsCreateMemoryStream");
    FnEdsCreateEvfImageRef pEdsCreateEvfImageRef = (FnEdsCreateEvfImageRef)GetProcAddress(hMod, "EdsCreateEvfImageRef");
    FnEdsDownloadEvfImage pEdsDownloadEvfImage = (FnEdsDownloadEvfImage)GetProcAddress(hMod, "EdsDownloadEvfImage");
    FnEdsGetPointer pEdsGetPointer = (FnEdsGetPointer)GetProcAddress(hMod, "EdsGetPointer");
    FnEdsGetLength pEdsGetLength = (FnEdsGetLength)GetProcAddress(hMod, "EdsGetLength");
    FnEdsRelease pEdsRelease = (FnEdsRelease)GetProcAddress(hMod, "EdsRelease");
    g_pSendCmd = (FnEdsSendCommand)GetProcAddress(hMod, "EdsSendCommand");

    EdsError err = pEdsInitializeSDK();
    if (err != 0) {
        fprintf(stderr, "[Canon Bridge] EdsInitializeSDK failed (0x%08X)\n", err);
        return 1;
    }

    EdsCameraListRef camList = NULL;
    err = pEdsGetCameraList(&camList);
    uint32_t count = 0;
    if (camList) pEdsGetChildCount(camList, &count);
    if (count == 0) {
        fprintf(stderr, "[Canon Bridge] No Canon cameras connected.\n");
        if (camList) pEdsRelease(camList);
        pEdsTerminateSDK();
        return 2;
    }

    pEdsGetChildAtIndex(camList, 0, (EdsBaseRef*)&g_cam);
    err = pEdsOpenSession(g_cam);
    if (err != 0) {
        fprintf(stderr, "[Canon Bridge] EdsOpenSession failed (0x%08X)\n", err);
        pEdsRelease(g_cam);
        pEdsRelease(camList);
        pEdsTerminateSDK();
        return 3;
    }

    // Enable EVF (live view) mode and direct output to PC
    uint32_t evfMode = 1;
    g_pSetProp(g_cam, 0x00000501, 0, sizeof(evfMode), &evfMode);
    uint32_t outDevice = 2; // PC
    g_pSetProp(g_cam, 0x00000500, 0, sizeof(outDevice), &outDevice);

    fprintf(stderr, "[Canon Bridge] Live View started on Canon EOS camera.\n");

    // Launch stdin listener thread
    HANDLE hThread = CreateThread(NULL, 0, StdinThread, NULL, 0, NULL);

    const uint32_t MAGIC = 0x46564501; // "EVF1"
    int notReadyCount = 0;

    while (g_running) {
        EdsStreamRef stream = NULL;
        err = pEdsCreateMemoryStream(0, &stream);
        if (err != 0) {
            Sleep(30);
            continue;
        }

        EdsEvfImageRef evfImg = NULL;
        err = pEdsCreateEvfImageRef(stream, &evfImg);
        if (err != 0) {
            pEdsRelease(stream);
            Sleep(30);
            continue;
        }

        err = pEdsDownloadEvfImage(g_cam, evfImg);
        if (err == 0) {
            notReadyCount = 0;
            void *pData = NULL;
            pEdsGetPointer(stream, &pData);
            uint64_t len = 0;
            pEdsGetLength(stream, &len);

            if (len > 0 && pData != NULL) {
                uint32_t uLen = (uint32_t)len;
                fwrite(&MAGIC, 4, 1, stdout);
                fwrite(&uLen, 4, 1, stdout);
                fwrite(pData, 1, uLen, stdout);
                fflush(stdout);
            }
        } else if (err == 0x00000061) { // EDS_ERR_OBJECT_NOTREADY
            notReadyCount++;
            if (notReadyCount > 50) {
                // Re-assert output device if sensor went to sleep
                g_pSetProp(g_cam, 0x00000500, 0, sizeof(outDevice), &outDevice);
                notReadyCount = 0;
            }
        }

        pEdsRelease(evfImg);
        pEdsRelease(stream);

        // Frame rate limiter: ~33ms per frame (approx 30 FPS)
        Sleep(30);
    }

    if (hThread) {
        CloseHandle(hThread);
    }

    // Clean shutdown
    uint32_t offDevice = 1; // TFT
    g_pSetProp(g_cam, 0x00000500, 0, sizeof(offDevice), &offDevice);
    pEdsCloseSession(g_cam);
    pEdsRelease(g_cam);
    pEdsRelease(camList);
    pEdsTerminateSDK();
    fprintf(stderr, "[Canon Bridge] Closed cleanly.\n");
    return 0;
}
