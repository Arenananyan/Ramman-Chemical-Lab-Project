// Apply the lab preset to USB Video Class cameras selected by location ID.
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/usb/IOUSBLib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static int property(io_service_t service, CFStringRef key) {
    CFTypeRef value = IORegistryEntryCreateCFProperty(service, key, NULL, 0);
    int result = -1;
    if (value && CFGetTypeID(value) == CFNumberGetTypeID())
        CFNumberGetValue(value, kCFNumberIntType, &result);
    if (value) CFRelease(value);
    return result;
}

static bool transfer(IOUSBInterfaceInterface220 **usb, UInt8 number,
                     UInt8 unit, UInt8 selector, UInt8 request, void *data, UInt16 size) {
    IOUSBDevRequestTO command = {
        .bmRequestType = request == 0x01 ? 0x21 : 0xa1,
        .bRequest = request,
        .wValue = selector << 8,
        .wIndex = (unit << 8) | number,
        .wLength = size,
        .pData = data,
        .noDataTimeout = 1000,
        .completionTimeout = 1000,
    };
    IOReturn result = (*usb)->ControlRequestTO(usb, 0, &command);
    if (result == kIOReturnSuccess && command.wLenDone == size) return true;
    fprintf(stderr, "UVC unit %u control %u request 0x%02x failed (0x%08x, %u bytes)\n",
            unit, selector, request, result, command.wLenDone);
    return false;
}

static bool preset(io_service_t service, unsigned location) {
    IOCFPlugInInterface **plugin = NULL;
    IOUSBInterfaceInterface220 **usb = NULL;
    SInt32 score = 0;
    IOReturn status = IOCreatePlugInInterfaceForService(
        service, kIOUSBInterfaceUserClientTypeID, kIOCFPlugInInterfaceID, &plugin, &score);
    if (status != kIOReturnSuccess || !plugin) {
        fprintf(stderr, "%08x: could not create USB interface (0x%08x)\n", location, status);
        return false;
    }
    HRESULT query = (*plugin)->QueryInterface(plugin, CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID220),
                                               (void **)&usb);
    IODestroyPlugInInterface(plugin);
    if (query || !usb) return false;

    UInt8 input = 0, processing = 0, number = 0;
    IOUSBDescriptorHeader *descriptor = NULL;
    while ((descriptor = (*usb)->FindNextAssociatedDescriptor(usb, descriptor, 0x24))) {
        const UInt8 *bytes = (const UInt8 *)descriptor;
        if (bytes[0] >= 6 && bytes[2] == 2 && bytes[4] == 1 && bytes[5] == 2)
            input = bytes[3]; // Camera input terminal, type 0x0201.
        if (bytes[0] >= 4 && bytes[2] == 5)
            processing = bytes[3];
    }

    bool ok = false;
    if (input && processing && (*usb)->GetInterfaceNumber(usb, &number) == kIOReturnSuccess) {
        UInt8 mode = 1;
        UInt32 exposure = 80;
        UInt16 gain = 0;
        ok = transfer(usb, number, input, 2, 0x01, &mode, sizeof(mode)) &&
             transfer(usb, number, input, 4, 0x01, &exposure, sizeof(exposure)) &&
             transfer(usb, number, processing, 4, 0x01, &gain, sizeof(gain)) &&
             transfer(usb, number, input, 2, 0x81, &mode, sizeof(mode)) &&
             transfer(usb, number, input, 4, 0x81, &exposure, sizeof(exposure)) &&
             transfer(usb, number, processing, 4, 0x81, &gain, sizeof(gain));
        if (ok) {
            printf("%08x: mode=%u exposure=%u gain=%u\n", location, mode, exposure, gain);
            ok = mode == 1 && exposure == 80 && gain == 0;
        }
    } else {
        fprintf(stderr, "%08x: UVC camera terminal or processing unit missing\n", location);
    }
    (*usb)->Release(usb);
    return ok;
}

int main(int argc, char **argv) {
    bool ok = argc > 1;
    for (int i = 1; i < argc; ++i) {
        unsigned location = (unsigned)strtoul(argv[i], NULL, 16);
        io_iterator_t interfaces = IO_OBJECT_NULL;
        if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOUSBHostInterface"),
                                         &interfaces) != kIOReturnSuccess) return 1;
        bool found = false;
        io_service_t service;
        while ((service = IOIteratorNext(interfaces))) {
            io_registry_entry_t device = IO_OBJECT_NULL;
            if (property(service, CFSTR("bInterfaceClass")) == 14 &&
                property(service, CFSTR("bInterfaceSubClass")) == 1 &&
                IORegistryEntryGetParentEntry(service, kIOServicePlane, &device) == kIOReturnSuccess) {
                found = (unsigned)property(device, CFSTR("locationID")) == location &&
                        property(device, CFSTR("idVendor")) == 0x09da &&
                        property(device, CFSTR("idProduct")) == 0x2695;
                IOObjectRelease(device);
            }
            if (found && !preset(service, location)) ok = false;
            IOObjectRelease(service);
            if (found) break;
        }
        IOObjectRelease(interfaces);
        if (!found) {
            fprintf(stderr, "A4Tech camera missing at USB location %08x\n", location);
            ok = false;
        }
    }
    return ok ? 0 : 1;
}
