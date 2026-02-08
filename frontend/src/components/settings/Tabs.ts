import {Bell, CreditCard, Database, Key, User} from "lucide-react";

export const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'data', label: 'Data Connections', icon: Database },
    { id: 'security', label: 'Security & API', icon: Key },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'billing', label: 'Billing', icon: CreditCard },
];
