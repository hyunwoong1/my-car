CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_no TEXT PRIMARY KEY,
    vehicle_type TEXT NOT NULL,
    registered_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS maintenance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_no TEXT NOT NULL REFERENCES vehicles(vehicle_no),
    date TEXT NOT NULL,
    item TEXT NOT NULL,
    cost INTEGER,
    next_due_date TEXT
);
