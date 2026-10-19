import CoreAudio
import Foundation

// Makes the named device the system input, unmuted at full volume. The call
// app takes whatever the system input is when the call starts, and macOS
// falls back to the built-in microphone on its own, so this runs every time.
let want = CommandLine.arguments.dropFirst().first ?? "BlackHole 2ch"
var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
var size: UInt32 = 0
AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size)
var ids = [AudioObjectID](repeating: 0, count: Int(size) / 4)
AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &ids)
for d in ids {
    var na = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal, mElement: 0)
    var name: Unmanaged<CFString>?
    var ns = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(d, &na, 0, nil, &ns, &name) == noErr,
          let n = name?.takeRetainedValue() as String?, n == want else { continue }
    for scope in [kAudioDevicePropertyScopeInput, kAudioDevicePropertyScopeOutput] {
        var mute: UInt32 = 0
        var ma = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyMute, mScope: scope, mElement: 0)
        AudioObjectSetPropertyData(d, &ma, 0, nil, 4, &mute)
        var vol: Float32 = 1
        var va = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyVolumeScalar, mScope: scope, mElement: 0)
        AudioObjectSetPropertyData(d, &va, 0, nil, 4, &vol)
    }
    var dev = d
    var da = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    let err = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &da, 0, nil, 4, &dev)
    print(err == noErr ? "system input: \(n)" : "could not set input (\(err))")
    exit(err == noErr ? 0 : 1)
}
print("no device named \(want)")
exit(1)
